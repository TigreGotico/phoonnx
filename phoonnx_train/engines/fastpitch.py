"""
FastPitch / SpeedySpeech training engine adapter.

Both architectures are configurations of the vendored ``ForwardTTS`` model
(``phoonnx_train/fastpitch/model.py`` — a self-contained, pure-torch port of
coqui-ai/TTS's ``TTS/tts/models/forward_tts.py``, MPL-2.0, see
``phoonnx_train/fastpitch/__init__.py`` for the license note):

- **FastPitch**: FFT-transformer encoder/decoder, pitch predictor on.
- **SpeedySpeech**: residual conv-BN encoder/decoder, no pitch predictor.

Durations are learned unsupervised (no external aligner needed) via the
vendored ``AlignmentNetwork`` + monotonic alignment search, same recipe as
upstream FastPitch/coqui.

The LightningModule / dataset / collate live in
``phoonnx_train.fastpitch.lightning``; heavy torch imports are deferred
until a model is actually built so the engine registry stays importable in
torch-free environments.

ONNX export follows the contract consumed by ``phoonnx.engines.fastpitch``
(``FastPitchAdapter``, which reuses ``MixerTTSAdapter``'s feed/parse logic):
inputs ``token_ids`` [B,T] + ``pace``/``pitch_mul``/``pitch_add`` [1]
(+ optional ``speaker`` [B] for multi-speaker) -> output ``mel_spec``
[B, 80, T_mel]. This mirrors
``scripts/conversion/coqui_fastpitch_export/export_fp.py``, which exports
pretrained Coqui FastPitch/SpeedySpeech checkpoints with the same I/O.
"""
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from phoonnx_train.engines.base import BaseTrainingEngine, TrainingEngineConfig

if TYPE_CHECKING:  # heavy import — only needed for type annotations
    import pytorch_lightning as pl

_LOG = logging.getLogger(__name__)

# ONNX opset used for export (matches coqui_fastpitch_export/export_fp.py)
OPSET_VERSION = 14

# Quality tier -> ForwardTTS hyper-param overrides.
# "variant" selects encoder/decoder family + pitch predictor (set via
# ``config.extra['variant']``, default per registered engine name).
_QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "x-low": {
        "hidden_channels": 128,
        "hidden_channels_ffn": 512,
        "encoder_num_layers": 4,
        "decoder_num_layers": 4,
        "num_heads": 1,
    },
    "medium": {
        "hidden_channels": 384,
        "hidden_channels_ffn": 1024,
        "encoder_num_layers": 6,
        "decoder_num_layers": 6,
        "num_heads": 1,
    },
    "high": {
        "hidden_channels": 512,
        "hidden_channels_ffn": 1536,
        "encoder_num_layers": 6,
        "decoder_num_layers": 6,
        "num_heads": 2,
    },
}


def resolve_module_kwargs(
    config: TrainingEngineConfig,
    default_variant: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Merge shared config + ``extra`` bag + call-site kwargs into the
    keyword arguments for ``ForwardTTSModule``.

    Torch-free (pure dict handling) so it is unit-testable without the
    training stack. ``extra['quality']``, if present, expands to the
    matching preset (unknown names fall back to ``medium``); explicit
    ``extra``/kwargs keys win over preset values, and ``variant`` defaults
    to *default_variant* when not set explicitly.
    """
    extra = dict(config.extra)
    preset_name = extra.pop("quality", None)
    merged: Dict[str, Any] = {}
    if preset_name is not None:
        if preset_name not in _QUALITY_PRESETS:
            _LOG.warning("unknown quality %r — falling back to 'medium'", preset_name)
            preset_name = "medium"
        merged.update(_QUALITY_PRESETS[preset_name])
    merged.update(extra)
    merged.update(kwargs)
    merged.setdefault("variant", default_variant)
    merged.update(
        num_symbols=config.num_symbols,
        num_speakers=config.num_speakers,
        sample_rate=config.sample_rate,
    )
    return merged


class ForwardTTSTrainingEngine(BaseTrainingEngine):
    """Training engine adapter for FastPitch / SpeedySpeech (ForwardTTS).

    Registered twice — as ``"fastpitch"`` and (via
    :class:`SpeedySpeechTrainingEngine`) as ``"speedyspeech"`` — the only
    difference being the default ``variant`` used when the caller doesn't
    set ``config.extra["variant"]`` explicitly.
    """

    default_variant = "fastpitch"

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    def create_model(
        self,
        config: TrainingEngineConfig,
        dataset_paths: List[Path],
        **kwargs: Any,
    ) -> "pl.LightningModule":
        from phoonnx_train.fastpitch.lightning import ForwardTTSModule

        return ForwardTTSModule(
            dataset=[str(p) for p in dataset_paths],
            **resolve_module_kwargs(config, self.default_variant, **kwargs),
        )

    def export_onnx(
        self,
        checkpoint_path: Path,
        config_path: Path,
        output_dir: Path,
        **kwargs: Any,
    ) -> Path:
        """Export a ForwardTTS checkpoint to ONNX.

        Contract matches ``coqui_fastpitch_export/export_fp.py`` /
        ``phoonnx.engines.fastpitch.FastPitchAdapter``: ``token_ids`` [B,T]
        + ``pace``/``pitch_mul``/``pitch_add`` [1] (+ optional ``speaker``
        [B]) -> ``mel_spec`` [B, 80, T_mel].
        """
        import json

        import torch

        from phoonnx_train.fastpitch.lightning import ForwardTTSModule
        from phoonnx_train.fastpitch.model import ForwardTTS
        from phoonnx_train.torch_compat import onnx_export_kwargs

        with open(config_path, "r", encoding="utf-8") as f:
            model_config: Dict[str, Any] = json.load(f)

        module: ForwardTTSModule = ForwardTTSModule.load_from_checkpoint(
            checkpoint_path, dataset=None, map_location="cpu",
        )
        model = module.model
        model.eval()

        num_symbols = model.args.num_chars
        num_speakers = model.args.num_speakers
        multispeaker = num_speakers > 1

        class _InferWrapper(torch.nn.Module):
            """pace / pitch_mul / pitch_add are graph inputs so the
            FastPitchAdapter's speed and pitch controls actually reach the
            model (traced as constants they would be silent no-ops)."""

            def __init__(self, m: ForwardTTS, multispeaker: bool):
                super().__init__()
                self.m = m
                self.multispeaker = multispeaker

            def forward(self, token_ids: torch.Tensor,
                       pace: torch.Tensor,
                       pitch_mul: torch.Tensor,
                       pitch_add: torch.Tensor,
                       speaker=None) -> torch.Tensor:
                sid = speaker.long() if self.multispeaker else None
                return self.m.inference(token_ids, speaker=sid, pace=pace,
                                        pitch_mul=pitch_mul, pitch_add=pitch_add)

        wrapper = _InferWrapper(model, multispeaker)
        wrapper.eval()

        dummy_ids = torch.randint(low=1, high=max(2, num_symbols - 1), size=(1, 30), dtype=torch.long)
        dummy_controls = (torch.ones(1), torch.ones(1), torch.zeros(1))
        control_names = ["pace", "pitch_mul", "pitch_add"]
        if multispeaker:
            # int32 to match the FastPitchAdapter/MixerTTSAdapter feed dtype;
            # cast to long inside the wrapper for the embedding lookup
            dummy_speaker = torch.zeros(1, dtype=torch.int32)
            dummy_args = (dummy_ids, *dummy_controls, dummy_speaker)
            input_names = ["token_ids", *control_names, "speaker"]
            dynamic_axes = {
                "token_ids": {0: "batch_size", 1: "phonemes"},
                "speaker": {0: "batch_size"},
                "mel_spec": {0: "batch_size", 2: "frame"},
            }
        else:
            dummy_args = (dummy_ids, *dummy_controls)
            input_names = ["token_ids", *control_names]
            dynamic_axes = {
                "token_ids": {0: "batch_size", 1: "phonemes"},
                "mel_spec": {0: "batch_size", 2: "frame"},
            }

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{checkpoint_path.stem}.onnx"

        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy_args,
                str(output_path),
                input_names=input_names,
                output_names=["mel_spec"],
                dynamic_axes=dynamic_axes,
                opset_version=OPSET_VERSION,
                **onnx_export_kwargs(),
            )

        try:
            import onnx as _onnx

            onnx_model = _onnx.load(str(output_path))
            del onnx_model.metadata_props[:]
            for key, value in {
                "model_type": module.hparams.variant,
                "engine": module.hparams.variant,
                "n_speakers": num_speakers,
                "n_vocab": num_symbols,
                "sample_rate": model_config.get("audio", {}).get("sample_rate", module.hparams.sample_rate),
                "alphabet": model_config.get("alphabet", ""),
                "phoneme_type": model_config.get("phoneme_type", ""),
                "phonemizer_model": model_config.get("phonemizer_model", ""),
                "phoneme_id_map": json.dumps(model_config.get("phoneme_id_map", {})),
                "has_espeak": model_config.get("phoneme_type", "") == "espeak",
            }.items():
                meta = onnx_model.metadata_props.add()
                meta.key = key
                meta.value = str(value)
            _onnx.save(onnx_model, str(output_path))
        except ImportError:
            _LOG.warning("onnx package not installed — skipping metadata")

        _LOG.info("Exported ForwardTTS (%s) ONNX model to %s", module.hparams.variant, output_path)
        return output_path

    def quality_presets(self) -> Dict[str, Dict[str, Any]]:
        return _QUALITY_PRESETS

    # ------------------------------------------------------------------
    # Optional
    # ------------------------------------------------------------------

    def extra_preprocess(
        self,
        utterance_audio_path: Path,
        cache_dir: Path,
        sample_rate: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Extract F0 (pitch) via pyworld; cached alongside the mel cache.

        Same contract as the OptiSpeech training engine's
        ``extra_preprocess`` (pyworld DIO + StoneMask). SpeedySpeech does
        not use pitch, but the field is harmless to compute/cache and lets
        the same preprocessed dataset be reused for either variant.
        """
        import numpy as np

        try:
            import librosa
            import pyworld as pw
        except ImportError:
            _LOG.warning(
                "pyworld/librosa not installed — skipping F0 extraction "
                "(FastPitch pitch predictor will train without a target; "
                "install phoonnx[train,train-fastpitch] for real pitch)."
            )
            return {}

        wav, sr = librosa.load(str(utterance_audio_path), sr=sample_rate, mono=True)
        wav_double = wav.astype(np.float64)
        f0, timeaxis = pw.dio(wav_double, sr)
        f0 = pw.stonemask(wav_double, f0, timeaxis, sr).astype(np.float32)

        cache_dir.mkdir(parents=True, exist_ok=True)
        stem = utterance_audio_path.stem
        f0_path = cache_dir / f"{stem}.f0.npy"
        np.save(f0_path, f0)

        return {"f0_path": str(f0_path)}

    def load_checkpoint(
        self,
        model: "pl.LightningModule",
        checkpoint_path: Path,
        **kwargs: Any,
    ) -> "pl.LightningModule":
        """Tolerant checkpoint load (skips shape-mismatched tensors, e.g.
        when fine-tuning onto a different phoneme inventory)."""
        import torch

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("state_dict", ckpt)
        model_state = model.state_dict()
        filtered = {}
        for k, v in state_dict.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered[k] = v
            elif k in model_state:
                _LOG.warning("Shape mismatch for %s: ckpt=%s model=%s — skipping",
                            k, v.shape, model_state[k].shape)
        model.load_state_dict(filtered, strict=False)
        _LOG.info("Loaded %d/%d parameters from checkpoint", len(filtered), len(model_state))
        return model


class SpeedySpeechTrainingEngine(ForwardTTSTrainingEngine):
    """Same ``ForwardTTS`` engine, defaulting to the SpeedySpeech variant
    (residual conv-BN encoder/decoder, no pitch predictor) when
    ``config.extra["variant"]`` isn't set explicitly."""

    default_variant = "speedyspeech"
