#!/usr/bin/env python3
import click
import logging
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from phoonnx_train.vits.lightning import VitsModel

_LOGGER = logging.getLogger("phoonnx_train.export_onnx")

# ONNX opset version
OPSET_VERSION = 15


# --- Utility Functions ---

def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    try:
        import onnx

        # Load the ONNX model
        model = onnx.load(filename)

        # Clear existing metadata and add new properties
        del model.metadata_props[:]

        for key, value in meta_data.items():
            meta = model.metadata_props.add()
            meta.key = key
            # Convert all values to string for ONNX metadata
            meta.value = str(value)

        onnx.save(model, filename)
        _LOGGER.info("Added %d metadata key/value pairs to ONNX model.", len(meta_data))

    except ImportError:
        _LOGGER.error("The 'onnx' package is required to add metadata. Please install it with 'pip install onnx'.")
    except Exception as e:
        _LOGGER.error("Failed to add metadata to ONNX file: %s", e)


def generate_tokens(config_path: Path):
    """Generates a tokens.txt file containing phoneme-to-id mapping from the config."""
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
    except Exception as e:
        _LOGGER.error("Failed to load config file at %s: %s", config_path, e)
        return

    id_map = config.get("phoneme_id_map")
    if not id_map:
        _LOGGER.error("Could not find 'phoneme_id_map' in the config file.")
        return

    tokens_path = Path("tokens.txt")
    with open(tokens_path, "w", encoding="utf-8") as f:
        # Sort by ID to ensure a consistent output order
        sorted_items = sorted(id_map.items(), key=lambda item: item[1])

        for s, i in sorted_items:
            # Skip newlines or other invalid tokens if present in map
            if s == "\n" or s == "":
                continue
            f.write(f"{s} {i}\n")

    _LOGGER.info("Generated tokens file at %s", tokens_path)


# --- Main Logic using Click ---
@click.command(help="Export a VITS model checkpoint to ONNX format.")
@click.argument(
    "checkpoint",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the PyTorch checkpoint file (*.ckpt)."
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=Path),
    default="config.json",
    help="Path to the model configuration JSON file. (Default: config.json)"
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default="model.onnx",
    help="Output path for the ONNX model. (Default: model.onnx)"
)
@click.option(
    "--dataset",
    type=str,
    default="unknown",
    help="Name of the dataset used for training, for metadata purposes. (Default: unknown)"
)
@click.option(
    "--generate-tokens",
    is_flag=True,
    help="Generate tokens.txt alongside the ONNX model. Some inference engines need this (eg. sherpa)"
)
@click.option(
    "--debug",
    is_flag=True,
    help="Print DEBUG messages to the console"
)
# Note: The function signature must match the argument/option names
def cli(
        checkpoint: Path,
        config: Path,
        output: Path,
        dataset: str,
        generate_tokens: bool,
        debug: bool
) -> None:
    """Main entry point"""
    torch.manual_seed(1234)

    # Configure logging based on the 'debug' option
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(f"Arguments: {checkpoint=}, {config=}, {output=}, {generate_tokens=}, {debug=}")

    # -------------------------------------------------------------------------
    # Paths and Setup

    # Create output directory if it doesn't exist
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load the phoonnx configuration
    try:
        with open(config, "r", encoding="utf-8") as f:
            model_config = json.load(f)
        _LOGGER.info("Loaded phoonnx config from %s", config)
    except Exception as e:
        _LOGGER.error("Error loading config file: %s", e)
        return

    if generate_tokens:
        # Generate the tokens.txt file
        generate_tokens(config)

    # -------------------------------------------------------------------------
    # Model Loading and Preparation
    try:
        model = VitsModel.load_from_checkpoint(
            checkpoint
        )
    except Exception as e:
        _LOGGER.error("Error loading model checkpoint: %s", e)
        return

    model_g = model.model_g
    num_symbols = model_g.n_vocab
    num_speakers = model_g.n_speakers

    # Inference only setup
    model_g.eval()

    with torch.no_grad():
        # Apply weight norm removal for inference mode
        model_g.dec.remove_weight_norm()
        _LOGGER.debug("Removed weight normalization from decoder.")

    # -------------------------------------------------------------------------
    # Define ONNX-compatible forward function

    def infer_forward(text, text_lengths, scales, sid=None):
        """Custom forward pass for ONNX export, simplifying the input scales."""
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]

        # model_g.infer returns (audio, attn, ids_slice, x_mask, z, z_mask, g)
        audio = model_g.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid,
        )[0].unsqueeze(1)  # [0] gets the audio tensor. unsqueeze(1) makes it [B, 1, T]

        return audio

    model_g.forward = infer_forward  # Replace the default forward with the inference one

    # -------------------------------------------------------------------------
    # Dummy Input Generation

    dummy_input_length = 50
    sequences = torch.randint(
        low=0, high=num_symbols, size=(1, dummy_input_length), dtype=torch.long
    )
    sequence_lengths = torch.LongTensor([sequences.size(1)])

    sid: Optional[torch.LongTensor] = None
    input_names = ["input", "input_lengths", "scales"]
    dynamic_axes_map = {
        "input": {0: "batch_size", 1: "phonemes"},
        "input_lengths": {0: "batch_size"},
        "output": {0: "batch_size", 1: "time"},
    }

    if num_speakers > 1:
        sid = torch.LongTensor([0])
        input_names.append("sid")
        dynamic_axes_map["sid"] = {0: "batch_size"}
        _LOGGER.debug("Multi-speaker model detected (n_speakers=%d). 'sid' included.", num_speakers)

    # noise, length, noise_w scales (hardcoded defaults)
    scales = torch.FloatTensor([0.667, 1.0, 0.8])
    dummy_input = (sequences, sequence_lengths, scales, sid)

    # -------------------------------------------------------------------------
    # Export

    _LOGGER.info("Starting ONNX export to %s (opset=%d)...", output, OPSET_VERSION)
    torch.onnx.export(
        model=model_g,
        args=dummy_input,
        f=str(output),
        verbose=False,
        opset_version=OPSET_VERSION,
        input_names=input_names,
        output_names=["output"],
        dynamic_axes=dynamic_axes_map,
    )

    _LOGGER.info("Successfully exported model to %s", output)

    # -------------------------------------------------------------------------
    # Add Metadata
    alphabet = config.get("alphabet", "")
    phoneme_type = config.get("phoneme_type", "")
    phonemizer_model = config.get("phonemizer_model", "")  # depends on phonemizer (eg. byt5)
    piper_compatible = alphabet == "ipa" and phoneme_type == "espeak"
    metadata_dict = {
        "model_type": "vits",
        "n_speakers": num_speakers,
        "n_vocab": num_symbols,
        "sample_rate": model_config["data"]["sampling_rate"],
        "dataset": dataset,
        "alphabet": alphabet,
        "phoneme_type": phoneme_type,
        "phonemizer_model": phonemizer_model,
        "phoneme_id_map": json.dumps(model_config["phoneme_id_map"],
        "has_espeak": phoneme_type == "espeak"
    }
    if piper_compatible:
        metadata_dict["comment"] = "piper"

    try:
        add_meta_data(str(output), metadata_dict)
    except Exception as e:
        _LOGGER.error(f"Failed to add metadata to exported model: ({e})")

    _LOGGER.info("Export complete.")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
