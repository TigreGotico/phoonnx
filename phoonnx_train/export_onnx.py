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
import click
import torch

from phoonnx_train.engines import get_engine, list_engines

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
def cli(
    checkpoint: Path,
    config: Path,
    output_dir: Path,
    engine: str,
    generate_tokens: bool,
    piper: bool,
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

    _LOGGER.info("Export complete: %s", onnx_path)


if __name__ == "__main__":
    cli()
