"""
Engine-aware ONNX export CLI.

Usage::

    python -m phoonnx_train.export_onnx model.ckpt \\
        --config config.json \\
        --engine vits \\
        --output-dir ./exported

Adding ``--engine optispeech`` (once registered) will transparently
use the OptiSpeech export procedure, metadata format, etc.
"""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import torch

from phoonnx_train.engines import get_engine, list_engines
from phoonnx_train.engines.vits import OPSET_VERSION, _write_piper_json, _write_tokens_txt
from phoonnx_train.vits.lightning import VitsModel

logging.basicConfig(level=logging.DEBUG)
_LOGGER = logging.getLogger("phoonnx_train.export_onnx")


def _validate_engine(ctx, param, value):
    available = list_engines()
    if value.lower() not in [e.lower() for e in available]:
        raise click.BadParameter(
            f"Unknown engine {value!r}. Choose from: {', '.join(available)}"
        )
    return value


@click.command(help="Export a model checkpoint to ONNX format.")
@click.argument(
    "checkpoint",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-c", "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the model configuration JSON file.",
)
@click.option(
    "-o", "--output-dir",
    type=click.Path(path_type=Path),
    default=Path(os.getcwd()),
    help="Output directory for the ONNX model.",
)
@click.option(
    "--engine",
    default="vits",
    type=str,
    callback=_validate_engine,
    help="TTS architecture used for training (default: vits).",
)
@click.option(
    "-t", "--generate-tokens",
    is_flag=True,
    help="Generate tokens.txt alongside the ONNX model.",
)
@click.option(
    "-p", "--piper",
    is_flag=True,
    help="Generate a piper-compatible JSON file.",
)
@click.option(
    "--disentangled-mode",
    type=click.Choice(["pre-encoded", "end-to-end"]),
    default=None,
    help="Export mode for disentangled models. 'pre-encoded' exports with pre-computed embedding inputs (recommended for edge). 'end-to-end' includes the reference encoders in the ONNX graph."
)
def cli(
    checkpoint: Path,
    config: Path,
    output_dir: Path,
    engine: str,
    generate_tokens: bool,
    piper: bool,
    disentangled_mode: Optional[str],
) -> None:
    """
    Main entry point for exporting a VITS model checkpoint to ONNX format.

    Args:
        checkpoint: Path to the PyTorch checkpoint file (*.ckpt).
        config: Path to the model configuration JSON file.
        output_dir: Output directory for the ONNX model and associated files.
        engine: TTS architecture used for training.
        generate_tokens: Flag to generate a tokens.txt file.
        piper: Flag to generate a piper compatible .json file.
        disentangled_mode: Export mode for disentangled models.
    """
    torch.manual_seed(1234)

    _LOGGER.debug(
        "Arguments: checkpoint=%s, config=%s, output_dir=%s, engine=%s, generate_tokens=%s, piper=%s, disentangled_mode=%s",
        checkpoint, config, output_dir, engine, generate_tokens, piper, disentangled_mode,
    )

    # -------------------------------------------------------------------------
    # Paths and Setup

    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve engine
    training_engine = get_engine(engine)
    _LOGGER.info("Using export engine: %s", engine)

    # Load config
    with open(config, "r", encoding="utf-8") as f:
        model_config: Dict[str, Any] = json.load(f)

    alphabet: str = model_config.get("alphabet", "")
    phoneme_type: str = model_config.get("phoneme_type", "")
    phonemizer_model: str = model_config.get("phonemizer_model", "")
    piper_compatible: bool = alphabet == "ipa" and phoneme_type == "espeak"

    sample_rate: int = model_config.get("audio", {}).get("sample_rate", 22050)
    phoneme_id_map: Dict[str, Any] = model_config.get("phoneme_id_map", {})

    if piper:
        if not piper_compatible:
            _LOGGER.warning("only models trained with ipa + espeak should be exported to piper. phonemization is not included in exported model.")
        piper_output_path = output_dir / f"{checkpoint.stem}.piper.json"
        _write_piper_json(model_config, piper_output_path)
        _LOGGER.info("Generated piper JSON: %s", piper_output_path)

    if generate_tokens:
        tokens_output_path = output_dir / f"{checkpoint.stem}.tokens.txt"
        _write_tokens_txt(phoneme_id_map, tokens_output_path)
        _LOGGER.info("Generated tokens.txt: %s", tokens_output_path)

    # -------------------------------------------------------------------------
    # Model Loading and Preparation
    try:
        model: VitsModel = VitsModel.load_from_checkpoint(
            checkpoint,
            dataset=None
        )
    except Exception as e:
        _LOGGER.error("Error loading model checkpoint %s: %s", checkpoint, e)
        return

    model_g: torch.nn.Module = model.model_g
    num_symbols: int = model_g.n_vocab
    num_speakers: int = model_g.n_speakers
    disentangled: bool = getattr(model_g, "disentangled", False)

    # Override disentangled mode detection if user specified it
    if disentangled_mode is not None:
        disentangled = True

    # Inference only setup
    model_g.eval()

    with torch.no_grad():
        # Apply weight norm removal for inference mode
        model_g.dec.remove_weight_norm()
        _LOGGER.debug("Removed weight normalization from decoder.")

    # -------------------------------------------------------------------------
    # Define ONNX-compatible forward function

    if disentangled and disentangled_mode == "pre-encoded":
        _LOGGER.info(
            "Disentangled pre-encoded mode requested; falling back to end-to-end "
            "export because pre-encoded embeddings require graph surgery. "
            "Encoders are lightweight (~50K params each) so this is acceptable."
        )
        disentangled_mode = "end-to-end"

    if disentangled and disentangled_mode == "end-to-end":
        def infer_forward(
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            scales: torch.Tensor,
            sid: Optional[torch.Tensor] = None,
            timbre_ref_mel: Optional[torch.Tensor] = None,
            artic_ref_mel: Optional[torch.Tensor] = None,
            prosody_ref_mel: Optional[torch.Tensor] = None,
            emotion_id: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            noise_scale: float = scales[0]
            length_scale: float = scales[1]
            noise_scale_w: float = scales[2]

            audio: torch.Tensor = model_g.infer(
                text,
                text_lengths,
                noise_scale=noise_scale,
                length_scale=length_scale,
                noise_scale_w=noise_scale_w,
                sid=sid,
                timbre_ref_mel=timbre_ref_mel,
                artic_ref_mel=artic_ref_mel,
                prosody_ref_mel=prosody_ref_mel,
                emotion_id=emotion_id,
            )[0].unsqueeze(1)  # [B, 1, T]
            return audio
    else:
        def infer_forward(text: torch.Tensor, text_lengths: torch.Tensor, scales: torch.Tensor, sid: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Custom forward pass for ONNX export, simplifying the input scales and
            returning only the audio tensor with shape [B, 1, T].
            """
            noise_scale: float = scales[0]
            length_scale: float = scales[1]
            noise_scale_w: float = scales[2]

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
    input_names: List[str] = ["input", "input_lengths", "scales"]
    dynamic_axes_map: Dict[str, Dict[int, str]] = {
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
    scales: torch.Tensor = torch.FloatTensor([0.667, 1.0, 0.8])

    dummy_input: List[torch.Tensor] = [sequences, sequence_lengths, scales]
    if sid is not None:
        dummy_input.append(sid)

    if disentangled and disentangled_mode == "end-to-end":
        # Add dummy reference mel inputs [B, n_mels, T_ref]
        n_mels = 80
        ref_len = 128
        dummy_timbre = torch.randn(1, n_mels, ref_len)
        dummy_artic = torch.randn(1, n_mels, ref_len)
        dummy_prosody = torch.randn(1, n_mels, ref_len)
        dummy_emotion = torch.LongTensor([0])

        input_names.extend([
            "timbre_ref_mel", "artic_ref_mel", "prosody_ref_mel", "emotion_id"
        ])
        dynamic_axes_map.update({
            "timbre_ref_mel": {0: "batch_size", 2: "ref_time"},
            "artic_ref_mel": {0: "batch_size", 2: "ref_time"},
            "prosody_ref_mel": {0: "batch_size", 2: "ref_time"},
            "emotion_id": {0: "batch_size"},
        })
        dummy_input.extend([dummy_timbre, dummy_artic, dummy_prosody, dummy_emotion])
        _LOGGER.debug("Disentangled end-to-end export: added timbre_ref_mel, artic_ref_mel, prosody_ref_mel, emotion_id inputs.")

    model_output = output_dir / f"{checkpoint.stem}.onnx"

    try:
        torch.onnx.export(
            model=model_g,
            args=tuple(dummy_input),
            f=str(model_output),
            verbose=False,
            opset_version=OPSET_VERSION,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes_map,
        )
        _LOGGER.info("Successfully exported model to %s", model_output)
    except Exception as e:
        _LOGGER.error("Failed during torch.onnx.export: %s", e)
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
        "has_espeak": phoneme_type == "espeak",
        "disentangled": str(disentangled),
        "disentangled_mode": disentangled_mode or "none",
    }
    if disentangled:
        metadata_dict["timbre_dim"] = getattr(model_g, "_timbre_dim", 0)
        metadata_dict["artic_dim"] = getattr(model_g, "_artic_dim", 0)
        metadata_dict["prosody_dim"] = getattr(model_g, "_prosody_dim", 0)
    if piper_compatible:
        metadata_dict["comment"] = "piper"

    try:
        import onnx as _onnx
        onnx_model = _onnx.load(str(model_output))
        del onnx_model.metadata_props[:]
        for key, value in metadata_dict.items():
            meta = onnx_model.metadata_props.add()
            meta.key = key
            meta.value = str(value)
        _onnx.save(onnx_model, str(model_output))
        _LOGGER.info("Added metadata to exported model.")
    except ImportError:
        _LOGGER.warning("onnx package not installed — skipping metadata")
    except Exception as e:
        _LOGGER.error("Failed to add metadata to exported model %s: %s", model_output, e)

    _LOGGER.info("Export complete: %s", model_output)


if __name__ == "__main__":
    cli()
