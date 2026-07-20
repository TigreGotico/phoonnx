"""Export a trained Vocos vocoder checkpoint to ONNX.

The exported graph takes ``mels`` ``[B, 80, T]`` (dynamic batch and T) and
emits the three STFT coefficient tensors — ``mag``, ``x`` (real), ``y``
(imaginary) — that the phoonnx runtime Vocos adapter
(:class:`phoonnx.engines.vocoders.vocos.VocosVocoder`) inverts to audio
with its overlap-add ISTFT.  A ``config.json`` with the STFT parameters is
written next to the model, and a torch-vs-onnxruntime waveform parity
check is printed.

Example::

    python -m phoonnx_train.export_vocos \\
        lightning_logs/version_0/checkpoints/epoch=99-step=1000.ckpt \\
        vocoder.onnx
"""
import inspect
import json
import logging
from pathlib import Path

import click
import numpy as np
import torch

from phoonnx_train.vocos.lightning import VocosTrainingModule

_LOGGER = logging.getLogger(__package__)

OPSET_VERSION = 17


def legacy_onnx_export(*args, **kwargs) -> None:
    """``torch.onnx.export`` pinned to the legacy TorchScript exporter.

    torch >= 2.6 flips ``torch.onnx.export`` to the TorchDynamo exporter by
    default, which imports the optional ``onnxscript`` package. This project
    does not depend on ``onnxscript`` (and CI does not install it), so force
    the legacy path whenever the running torch exposes the ``dynamo`` kwarg;
    older torch (< 2.5) has no such kwarg and is already legacy-only.
    """
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs.setdefault("dynamo", False)
    torch.onnx.export(*args, **kwargs)


class _CoefficientWrapper(torch.nn.Module):
    """mel → (mag, x, y), the layout the runtime Vocos adapter expects."""

    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, mels: torch.Tensor):
        return self.generator.coefficients(mels)


@click.command()
@click.argument("checkpoint", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
@click.option("--seconds", default=2.0, type=float,
              help="Length of the random mel used for the parity check")
def main(checkpoint: str, output: str, seconds: float):
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(0)

    model = VocosTrainingModule.load_from_checkpoint(
        checkpoint, map_location="cpu", audio_files=[]
    )
    model.eval()
    generator = model.generator
    mel = model.mel

    wrapper = _CoefficientWrapper(generator)
    num_frames = int(seconds * mel.sample_rate) // mel.hop_length
    dummy = torch.randn(1, mel.n_mels, num_frames) * 2.0 - 6.0  # log-mel range

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_onnx_export(
        wrapper,
        dummy,
        str(output_path),
        opset_version=OPSET_VERSION,
        input_names=["mels"],
        output_names=["mag", "x", "y"],
        dynamic_axes={
            "mels": {0: "batch", 2: "time"},
            "mag": {0: "batch", 2: "time"},
            "x": {0: "batch", 2: "time"},
            "y": {0: "batch", 2: "time"},
        },
    )
    _LOGGER.info("Exported ONNX vocoder: %s", output_path)

    config = {
        "vocoder_type": "vocos",
        "n_fft": mel.n_fft,
        "hop_length": mel.hop_length,
        "sample_rate": mel.sample_rate,
        "num_mels": mel.n_mels,
        "fmin": mel.fmin,
        "fmax": mel.fmax,
    }
    config_path = output_path.with_suffix(".onnx.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    _LOGGER.info("Wrote vocoder config: %s", config_path)

    # ------------------------------------------------------------------
    # Parity: torch generator (its own ISTFT) vs the runtime adapter
    # running the exported graph + numpy overlap-add ISTFT.
    # ------------------------------------------------------------------
    from phoonnx.engines.vocoders import build_vocoder

    with torch.no_grad():
        audio_torch = generator(dummy).numpy()

    vocoder = build_vocoder(
        model_path=str(output_path), vocoder_type="vocos", config=config
    )
    audio_onnx = vocoder.mel_to_audio(dummy.numpy().astype(np.float32))

    n = min(audio_torch.shape[-1], audio_onnx.shape[-1])
    diff = float(np.max(np.abs(audio_torch.squeeze()[:n] - audio_onnx.squeeze()[:n])))
    print(f"Parity torch vs onnxruntime (max abs diff): {diff:.3e}")
    if diff >= 1e-3:
        raise SystemExit(f"Parity check FAILED: {diff:.3e} >= 1e-3")
    print("Parity check passed (< 1e-3)")


if __name__ == "__main__":
    main()
