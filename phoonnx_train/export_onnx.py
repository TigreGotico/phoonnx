#!/usr/bin/env python3
import click
import logging
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
from phoonnx_train.vits.lightning import VitsModel
from phoonnx.version import VERSION_STR

# Basic logging configuration
logging.basicConfig(level=logging.DEBUG)
_LOGGER = logging.getLogger("phoonnx_train.export_onnx")

# ONNX opset version
OPSET_VERSION = 15


# --- Utility Functions ---

def add_meta_data(filename: Path, meta_data: Dict[str, Any]) -> None:
    """
    Add meta data to an ONNX model. The file is modified in-place.

    Args:
      filename:
        Path to the ONNX model file to be changed.
      meta_data:
        Key-value pairs to be stored as metadata. Values will be converted to strings.
    """
    try:
        import onnx

        # Load the ONNX model
        model = onnx.load(str(filename))

        # Clear existing metadata and add new properties
        del model.metadata_props[:]

        for key, value in meta_data.items():
            meta = model.metadata_props.add()
            meta.key = key
            # Convert all values to string for ONNX metadata
            meta.value = str(value)

        onnx.save(model, str(filename))
        _LOGGER.info(f"Added {len(meta_data)} metadata key/value pairs to ONNX model: {filename}")

    except ImportError:
        _LOGGER.error("The 'onnx' package is required to add metadata. Please install it with 'pip install onnx'.")
    except Exception as e:
        _LOGGER.error(f"Failed to add metadata to ONNX file {filename}: {e}")


def export_tokens(config_path: Path, output_path: Path = Path("tokens.txt")) -> None:
    """
    Generates a tokens.txt file containing phoneme-to-id mapping from the model configuration.

    The format is: `<phoneme> <id>` per line.

    Args:
        config_path: Path to the model configuration JSON file.
        output_path: Path to save the resulting tokens.txt file.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config: Dict[str, Any] = json.load(file)
    except Exception as e:
        _LOGGER.error(f"Failed to load config file at {config_path}: {e}")
        return

    id_map: Optional[Dict[str, int]] = config.get("phoneme_id_map")
    if not id_map:
        _LOGGER.error("Could not find 'phoneme_id_map' in the config file.")
        return

    tokens_path = output_path
    try:
        with open(tokens_path, "w", encoding="utf-8") as f:
            # Sort by ID to ensure a consistent output order
            # The type hint for sorted_items is a list of tuples: List[Tuple[str, int]]
            sorted_items: list[Tuple[str, int]] = sorted(id_map.items(), key=lambda item: item[1])

            for s, i in sorted_items:
                # Skip newlines or other invalid tokens if present in map
                if s == "\n" or s == "":
                    continue
                f.write(f"{s} {i}\n")

        _LOGGER.info(f"Generated tokens file at {tokens_path}")
    except Exception as e:
        _LOGGER.error(f"Failed to write tokens file to {tokens_path}: {e}")


def convert_to_piper(config_path: Path, output_path: Path = Path("piper.json")) -> None:
    """
    Generates a Piper compatible JSON configuration file from the VITS model configuration.

    This function currently serves as a placeholder for full Piper conversion logic.

    Args:
        config_path: Path to the VITS model configuration JSON file.
        output_path: Path to save the resulting Piper JSON file.
    """

    with open(config_path, "r", encoding="utf-8") as file:
        config: Dict[str, Any] = json.load(file)

    piper_config = {
        "phoneme_type": "espeak" if config.get("phoneme_type", "") == "espeak" else "raw",
        "phoneme_map": {},
        "audio": config.get("audio", {}),
        "inference": config.get("inference", {}),
        "phoneme_id_map": {k: [v] for k, v in config.get("phoneme_id_map", {}).items()},
        "espeak": {
            "voice": config.get("lang_code", "")
        },
        "language": {
            "code": config.get("lang_code", "")
        },
        "num_symbols": config.get("num_symbols", 256),
        "num_speakers": config.get("num_speakers", 1),
        "speaker_id_map": {},
        "piper_version": f"phoonnx-" + config.get("phoonnx_version", "0.0.0")
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(piper_config, f, indent=4, ensure_ascii=False)


# --- Main Logic using Click ---
@click.command(help="Export a VITS model checkpoint to ONNX format.")
@click.argument(
    "checkpoint",
    type=click.Path(exists=True, path_type=Path),
  #  help="Path to the PyTorch checkpoint file (*.ckpt)."
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the model configuration JSON file."
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path(os.getcwd()), # Set default to current working directory
    help="Output directory for the ONNX model. (Default: current directory)"
)
@click.option(
    "-t",
    "--generate-tokens",
    is_flag=True,
    help="Generate tokens.txt alongside the ONNX model. Some inference engines need this (eg. sherpa)"
)
@click.option(
    "-p",
    "--piper",
    is_flag=True,
    help="Generate a piper compatible .json file alongside the ONNX model."
)
def cli(
        checkpoint: Path,
        config: Path,
        output_dir: Path,
        generate_tokens: bool,
        piper: bool,
) -> None:
    """
    Main entry point for exporting a VITS model checkpoint to ONNX format.

    Args:
        checkpoint: Path to the PyTorch checkpoint file (*.ckpt).
        config: Path to the model configuration JSON file.
        output_dir: Output directory for the ONNX model and associated files.
        generate_tokens: Flag to generate a tokens.txt file.
        piper: Flag to generate a piper compatible .json file.
    """
    torch.manual_seed(1234)

    _LOGGER.debug(f"Arguments: {checkpoint=}, {config=}, {output_dir=}, {generate_tokens=}, {piper=}")

    # -------------------------------------------------------------------------
    # Paths and Setup

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    _LOGGER.debug(f"Output directory ensured: {output_dir}")

    # Load the phoonnx configuration
    try:
        with open(config, "r", encoding="utf-8") as f:
            model_config: Dict[str, Any] = json.load(f)
        _LOGGER.info(f"Loaded phoonnx config from {config}")
    except Exception as e:
        _LOGGER.error(f"Error loading config file {config}: {e}")
        return


    alphabet: str = model_config.get("alphabet", "")
    phoneme_type: str = model_config.get("phoneme_type", "")
    phonemizer_model: str = model_config.get("phonemizer_model", "")  # depends on phonemizer (eg. byt5)
    piper_compatible: bool = alphabet == "ipa" and phoneme_type == "espeak"

    # Ensure mandatory keys exist before accessing
    sample_rate: int = model_config.get("audio", {}).get("sample_rate", 22050)
    phoneme_id_map: Dict[str, int] = model_config.get("phoneme_id_map", {})

    if piper:
        if not piper_compatible:
            _LOGGER.warning("only models trained with ipa + espeak should be exported to piper. phonemization is not included in exported model.")
        # Generate the piper.json file
        piper_output_path = output_dir / f"{checkpoint.name}.piper.json"
        convert_to_piper(config, piper_output_path)

    if generate_tokens:
        # Generate the tokens.txt file
        tokens_output_path = output_dir / f"{checkpoint.name}.tokens.txt"
        export_tokens(config, tokens_output_path)

    # -------------------------------------------------------------------------
    # Model Loading and Preparation
    try:
        model: VitsModel = VitsModel.load_from_checkpoint(
            checkpoint,
            dataset=None
        )
    except Exception as e:
        _LOGGER.error(f"Error loading model checkpoint {checkpoint}: {e}")
        return

    model_g: torch.nn.Module = model.model_g
    num_symbols: int = model_g.n_vocab
    num_speakers: int = model_g.n_speakers

    # Inference only setup
    model_g.eval()

    with torch.no_grad():
        # Apply weight norm removal for inference mode
        model_g.dec.remove_weight_norm()
        _LOGGER.debug("Removed weight normalization from decoder.")

    # -------------------------------------------------------------------------
    # Define ONNX-compatible forward function

    def infer_forward(text: torch.Tensor, text_lengths: torch.Tensor, scales: torch.Tensor, sid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Custom forward pass for ONNX export, simplifying the input scales and
        returning only the audio tensor with shape [B, 1, T].

        Args:
            text: Input phoneme sequence tensor, shape [B, T_in].
            text_lengths: Tensor of sequence lengths, shape [B].
            scales: Tensor containing [noise_scale, length_scale, noise_scale_w], shape [3].
            sid: Optional speaker ID tensor, shape [B], for multi-speaker models.

        Returns:
            Generated audio tensor, shape [B, 1, T_out].
        """
        noise_scale: float = scales[0]
        length_scale: float = scales[1]
        noise_scale_w: float = scales[2]

        # model_g.infer returns a tuple: (audio, attn, ids_slice, x_mask, z, z_mask, g)
        audio: torch.Tensor = model_g.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid,
        )[0].unsqueeze(1)  # [0] gets the audio tensor. unsqueeze(1) makes it [B, 1, T]

        return audio

    # Replace the default forward with the inference one for ONNX export
    model_g.forward = infer_forward

    # -------------------------------------------------------------------------
    # Dummy Input Generation

    dummy_input_length: int = 50
    sequences: torch.Tensor = torch.randint(
        low=0, high=num_symbols, size=(1, dummy_input_length), dtype=torch.long
    )
    sequence_lengths: torch.Tensor = torch.LongTensor([sequences.size(1)])

    sid: Optional[torch.LongTensor] = None
    input_names: list[str] = ["input", "input_lengths", "scales"]
    dynamic_axes_map: Dict[str, Dict[int, str]] = {
        "input": {0: "batch_size", 1: "phonemes"},
        "input_lengths": {0: "batch_size"},
        "output": {0: "batch_size", 1: "time"},
    }

    if num_speakers > 1:
        sid = torch.LongTensor([0])
        input_names.append("sid")
        dynamic_axes_map["sid"] = {0: "batch_size"}
        _LOGGER.debug(f"Multi-speaker model detected (n_speakers={num_speakers}). 'sid' included.")

    # noise, length, noise_w scales (hardcoded defaults)
    scales: torch.Tensor = torch.FloatTensor([0.667, 1.0, 0.8])
    dummy_input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.LongTensor]] = (
        sequences, sequence_lengths, scales, sid
    )

    # -------------------------------------------------------------------------
    # Export
    model_output: Path = output_dir / f"{checkpoint.name}.onnx"
    _LOGGER.info(f"Starting ONNX export to {model_output} (opset={OPSET_VERSION})...")

    try:
        torch.onnx.export(
            model=model_g,
            args=dummy_input,
            f=str(model_output),
            verbose=False,
            opset_version=OPSET_VERSION,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes_map,
        )
        _LOGGER.info(f"Successfully exported model to {model_output}")
    except Exception as e:
        _LOGGER.error(f"Failed during torch.onnx.export: {e}")
        return

    # -------------------------------------------------------------------------
    # Add Metadata
    metadata_dict: Dict[str, Any] = {
        "model_type": "vits",
        "n_speakers": num_speakers,
        "n_vocab": num_symbols,
        "sample_rate": sample_rate,
        "alphabet": alphabet,
        "phoneme_type": phoneme_type,
        "phonemizer_model": phonemizer_model,
        "phoneme_id_map": json.dumps(phoneme_id_map),
        "has_espeak": phoneme_type == "espeak"
    }
    if piper_compatible:
        metadata_dict["comment"] = "piper"

    try:
        add_meta_data(model_output, metadata_dict)
    except Exception as e:
        _LOGGER.error(f"Failed to add metadata to exported model {model_output}: {e}")

    _LOGGER.info("Export complete.")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    cli()