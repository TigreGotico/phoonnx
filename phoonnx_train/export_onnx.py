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
import logging
import os
from pathlib import Path
from typing import Optional, Set
import click
import torch

from phoonnx_train.engines import get_engine, list_engines

logging.basicConfig(level=logging.DEBUG)
_LOGGER = logging.getLogger("phoonnx_train.export_onnx")


def add_phoneme_alignment_output(
    model_path: Path,
    output_path: Optional[Path] = None,
    tensor_name: str = "autodetect",
) -> None:
    """Expose a per-phoneme duration tensor as an extra ONNX graph output.

    Standard VITS exports emit only the audio tensor; the phoneme-alignment
    feature (see ``docs/alignment.md``) needs the duration-predictor's
    per-phoneme frame counts surfaced as a named output. This walks the graph,
    finds the (single) ``Ceil`` node output — the rounded durations — and marks
    it as a graph output.

    This may break third-party frameworks (e.g. piper) that expect a single
    output tensor, so keep a separate copy for standard TTS use.

    The actual graph surgery (locating the tensor + promoting it to an
    output) is pure-``onnx`` and lives in ``phoonnx.onnx_surgery`` so it can
    also be used at runtime, without a ``torch`` import, by
    ``phoonnx.voice.TTSVoice`` for models that were exported without this
    flag (see ``docs/alignment.md``, "runtime alignment for models exported
    without the flag").

    Args:
        model_path: Path to the input ONNX model file.
        output_path: Where to save the modified model (defaults to overwriting).
        tensor_name: Tensor to mark as an output. ``"autodetect"`` looks for a
            unique ``Ceil`` node output.
    """
    import onnx

    from phoonnx.onnx_surgery import add_phoneme_alignment_output as _surgery
    from phoonnx.onnx_surgery import find_duration_tensor

    output_path = output_path or model_path

    try:
        model: "onnx.ModelProto" = onnx.load(model_path)
    except Exception as e:
        _LOGGER.fatal(f"Failed to load ONNX model from {model_path}: {e}")
        return

    tensor = find_duration_tensor(model, tensor_name=tensor_name)
    if tensor is None:
        if tensor_name != "autodetect":
            _LOGGER.fatal(f"Tensor '{tensor_name}' not found in {model_path}. Use --tensor-name manually.")
            return
        ceil_tensor_names: Set[str] = {
            o for node in model.graph.node if node.op_type == "Ceil" for o in node.output
        }
        if not ceil_tensor_names:
            _LOGGER.fatal(f"No ceil tensors detected in {model_path}. Use --tensor-name manually.")
        else:
            names_str = ', '.join(sorted(ceil_tensor_names))
            _LOGGER.fatal(
                f"Multiple ceil tensors detected in {model_path}. Use --tensor-name manually: {names_str}"
            )
        return

    if tensor_name == "autodetect":
        _LOGGER.info(f"Detected tensor name: {tensor}")

    if any(output.name == tensor for output in model.graph.output):
        _LOGGER.fatal(f"Tensor '{tensor}' is already marked as output. Aborting.")
        return

    _surgery(model, tensor_name=tensor)

    try:
        onnx.save(model, output_path)
    except Exception as e:
        _LOGGER.fatal(f"Failed to save modified ONNX model to {output_path}: {e}")
        return

    _LOGGER.info(f"Successfully wrote modified model with new output '{tensor}' to {output_path}")


def _validate_engine(ctx, param, value):
    available = list_engines()
    if value.lower() not in [e.lower() for e in available]:
        raise click.BadParameter(
            f"Unknown engine {value!r}. Choose from: {', '.join(available)}"
        )
    return value.lower()


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
    "-a", "--add-phoneme-alignment",
    is_flag=True,
    help="Expose the phoneme duration tensor as an extra ONNX output for "
         "alignment (VITS only; may break piper-style single-output consumers).",
)
def cli(
    checkpoint: Path,
    config: Path,
    output_dir: Path,
    engine: str,
    generate_tokens: bool,
    piper: bool,
    add_phoneme_alignment: bool,
) -> None:
    torch.manual_seed(1234)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve engine
    training_engine = get_engine(engine)
    _LOGGER.info("Using export engine: %s", engine)

    # Delegate the full export to the engine
    onnx_path = training_engine.export_onnx(
        checkpoint_path=checkpoint,
        config_path=config,
        output_dir=output_dir,
        generate_tokens=generate_tokens,
        piper=piper,
    )

    if add_phoneme_alignment:
        _LOGGER.info("Adding phoneme-alignment output to %s", onnx_path)
        add_phoneme_alignment_output(Path(onnx_path))

    _LOGGER.info("Export complete: %s", onnx_path)


if __name__ == "__main__":
    cli()
