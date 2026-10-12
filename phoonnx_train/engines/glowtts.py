"""
GlowTTS training engine adapter.

Wraps :mod:`phoonnx_train.glowtts` (a self-contained, pure-torch
reimplementation of the GlowTTS architecture — see
``phoonnx_train/glowtts/__init__.py`` for the full provenance note) behind
the ``BaseTrainingEngine`` interface so the shared CLI tools can drive it.

GlowTTS is **two-stage**: this engine trains and exports only the
text -> mel model. The waveform vocoder is a *separate* artifact, reused
unchanged from :mod:`phoonnx.engines.vocoders` (HiFi-GAN, Vocos, Griffin-Lim,
...) at inference time by ``phoonnx.engines.glowtts.GlowTTSAdapter`` — this
engine's ``export_onnx`` never touches vocoder weights.

The exported ONNX graph's input/output contract (``input`` /
``input_lengths`` / ``scales`` -> ``[B, n_mels, T]`` mel) exactly matches
what ``phoonnx/engines/glowtts.py``'s ``GlowTTSAdapter`` expects, as read
directly from that file.
"""
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from phoonnx_train.engines.base import BaseTrainingEngine, TrainingEngineConfig

if TYPE_CHECKING:  # heavy import — only needed for type annotations
    import pytorch_lightning as pl

_LOG = logging.getLogger(__name__)

# ONNX opset used for export
OPSET_VERSION = 15

# Quality tier -> GlowTTS hyper-param overrides.
# Roughly mirrors coqui's glow_tts tiers (encoder/decoder width scaling
# consistent with VITS's own x-low/medium/high split in vits.py); exact
# upstream coqui dimensions were not independently verified — these are
# reasonable defaults for a paper-faithful GlowTTS, not a confirmed match.
_QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "x-low": {
        "hidden_channels": 96,
        "filter_channels": 384,
        "filter_channels_dp": 128,
        "n_heads": 2,
        "n_layers": 4,
        "dec_hidden_channels": 96,
        "dec_n_blocks": 8,
        "dec_n_layers": 3,
    },
    "medium": {
        "hidden_channels": 192,
        "filter_channels": 768,
        "filter_channels_dp": 256,
        "n_heads": 2,
        "n_layers": 6,
        "dec_hidden_channels": 192,
        "dec_n_blocks": 12,
        "dec_n_layers": 4,
    },
    "high": {
        "hidden_channels": 256,
        "filter_channels": 1024,
        "filter_channels_dp": 384,
        "n_heads": 4,
        "n_layers": 8,
        "dec_hidden_channels": 256,
        "dec_n_blocks": 12,
        "dec_n_layers": 4,
    },
}


class GlowTTSTrainingEngine(BaseTrainingEngine):

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    def create_model(
        self,
        config: TrainingEngineConfig,
        dataset_paths: List[Path],
        **kwargs: Any,
    ) -> "pl.LightningModule":
        """Build a GlowTTSModel LightningModule from *config*."""
        from phoonnx_train.glowtts.lightning import GlowTTSModel

        return GlowTTSModel(
            num_symbols=config.num_symbols,
            num_speakers=config.num_speakers,
            sample_rate=config.sample_rate,
            dataset=[str(p) for p in dataset_paths],
            **config.extra,
            **kwargs,
        )

    def export_onnx(
        self,
        checkpoint_path: Path,
        config_path: Path,
        output_dir: Path,
        **kwargs: Any,
    ) -> Path:
        """
        Export a GlowTTS checkpoint to ONNX (mel model only).

        The exported graph takes ``input`` (phoneme ids), ``input_lengths``,
        and ``scales`` ([noise_scale, length_scale]) — optionally ``sid`` for
        multi-speaker — and returns a ``[B, n_mels, T]`` mel spectrogram, per
        ``phoonnx.engines.glowtts.GlowTTSAdapter``. The vocoder that turns
        this mel into audio is a **separate** ONNX artifact (HiFi-GAN /
        Vocos / Griffin-Lim, see ``phoonnx/engines/vocoders/``) — it is not
        produced by this method.
        """
        # validate the inputs before touching the heavy imports so a bad
        # path fails fast (and identically) with or without torch installed
        with open(config_path, "r", encoding="utf-8") as f:
            model_config: Dict[str, Any] = json.load(f)
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

        import torch

        from phoonnx_train.glowtts.lightning import GlowTTSModel
        from phoonnx_train.torch_compat import onnx_export_kwargs

        sample_rate = model_config.get("audio", {}).get("sample_rate", 22050)
        phoneme_id_map = model_config.get("phoneme_id_map", {})
        alphabet = model_config.get("alphabet", "")
        phoneme_type = model_config.get("phoneme_type", "")
        phonemizer_model = model_config.get("phonemizer_model", "")

        model: GlowTTSModel = GlowTTSModel.load_from_checkpoint(
            checkpoint_path, dataset=None, map_location="cpu",
        )
        model_g = model.model_g
        model_g.eval()
        # torch.inverse (used by the flow's invertible 1x1 conv reverse pass)
        # has no ONNX symbolic — precompute and cache the inverse weights so
        # the traced reverse graph only contains a plain conv2d.
        model_g.decoder.store_inverse()

        num_symbols = model_g.n_vocab
        num_speakers = model_g.n_speakers
        n_mels = model_g.n_mels

        def infer_forward(input, input_lengths, scales, sid=None):
            noise_scale = scales[0]
            length_scale = scales[1]
            mel, _mel_lengths = model_g.infer(
                input, input_lengths,
                noise_scale=noise_scale, length_scale=length_scale, sid=sid,
            )
            return mel

        model_g.forward = infer_forward

        sequences = torch.randint(low=1, high=num_symbols, size=(1, 50), dtype=torch.long)
        sequence_lengths = torch.LongTensor([sequences.size(1)])
        scales = torch.FloatTensor([0.667, 1.0])
        sid = torch.LongTensor([0]) if num_speakers > 1 else None

        input_names = ["input", "input_lengths", "scales"]
        output_names = ["mel"]
        dynamic_axes = {
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "mel": {0: "batch_size", 2: "time"},
        }
        dummy_input = (sequences, sequence_lengths, scales)
        if sid is not None:
            input_names.append("sid")
            dynamic_axes["sid"] = {0: "batch_size"}
            dummy_input = (*dummy_input, sid)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{checkpoint_path.stem}.onnx"
        with torch.no_grad():
            # onnx_export_kwargs() forces the TorchScript exporter on
            # torch>=2.5 — the dynamo exporter cannot trace GlowTTS's
            # data-dependent control flow (length-derived squeeze)
            torch.onnx.export(
                model=model_g,
                args=dummy_input,
                f=str(output_path),
                verbose=False,
                opset_version=OPSET_VERSION,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                **onnx_export_kwargs(),
            )

        try:
            import onnx as _onnx

            onnx_model = _onnx.load(str(output_path))

            # The mel output's channel axis (dim 1) is a compile-time
            # constant (= n_mels) — it never varies with input, only batch
            # and time do. The exporter's shape inference nonetheless
            # stamps it with a symbolic name (derived from the reshape ops
            # in the flow decoder's squeeze/unsqueeze), because the value
            # is only *provably* constant, not literally folded during
            # tracing. Pin it back to the concrete n_mels so
            # GlowTTSAdapter.detect()'s ``o.shape[1] in (80, 0)`` heuristic
            # (and any consumer relying on a concrete mel-channel count)
            # sees the real, fixed value.
            for output in onnx_model.graph.output:
                if output.name == "mel":
                    dim1 = output.type.tensor_type.shape.dim[1]
                    dim1.ClearField("dim_param")
                    dim1.dim_value = n_mels

            del onnx_model.metadata_props[:]
            for key, value in {
                "model_type": "glow_tts",
                "engine": "glowtts",
                "mel_fmin": model.hparams.mel_fmin,
                "mel_fmax": model.hparams.mel_fmax,
                "n_speakers": num_speakers,
                "n_vocab": num_symbols,
                "n_mels": n_mels,
                "sample_rate": sample_rate,
                "alphabet": alphabet,
                "phoneme_type": phoneme_type,
                "phonemizer_model": phonemizer_model,
                "phoneme_id_map": json.dumps(phoneme_id_map),
                "has_espeak": phoneme_type == "espeak",
            }.items():
                meta = onnx_model.metadata_props.add()
                meta.key = key
                meta.value = str(value)
            _onnx.save(onnx_model, str(output_path))
        except ImportError:
            _LOG.warning("onnx package not installed — skipping metadata")

        _LOG.info(
            "Exported GlowTTS mel-model ONNX to %s (a separate vocoder ONNX "
            "is required for synthesis — see phoonnx/engines/vocoders/)",
            output_path,
        )
        return output_path

    def quality_presets(self) -> Dict[str, Dict[str, Any]]:
        """Return GlowTTS quality tier -> hyper-parameter overrides."""
        return _QUALITY_PRESETS

    # ------------------------------------------------------------------
    # Optional
    # ------------------------------------------------------------------

    def extra_cli_options(self) -> List[Any]:
        """GlowTTS has no engine-specific CLI options beyond the shared ones."""
        return []
