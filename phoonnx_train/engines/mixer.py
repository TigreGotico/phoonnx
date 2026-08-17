"""
Mixer-TTS training engine adapter.

Mixer-TTS (Tatanov et al., 2021, https://arxiv.org/abs/2110.03584) is a
non-autoregressive MLP-Mixer acoustic model: token/channel-mixing
encoder/decoder blocks, FastPitch-style duration/pitch (and here also
energy) predictors, and an unsupervised AlignmentEncoder trained with the
ForwardSum + binarization losses of the one-TTS-alignment recipe
(https://arxiv.org/abs/2108.10447).

The vendored model lives in ``phoonnx_train/mixertts/models`` — a
self-contained, pure-torch port of NVIDIA NeMo's implementation
(Apache-2.0, nemo/collections/tts/models/mixer_tts.py @ 7256db1) with the
speaker/emotion/energy conditioning and optional LSGAN mel-patch
refinement from nipponjo/tts-arabic-pytorch (MIT). The LightningModule /
dataset / collate live in ``phoonnx_train.mixertts.lightning``; heavy
torch imports are deferred until a model is actually built so the engine
registry stays importable in torch-free environments.

Pitch (F0) preprocessing is shared with the FastPitch engine (``librosa.pyin``
at a hop matched to ``hop / sample_rate`` seconds on the same
trimmed/normalized cached audio the mels come from, length-reconciled
to the mel frame count) — the ``<utterance>.f0-<method>.npy`` sidecars and
``pitch_stats.json`` are engine-compatible.

ONNX export follows the contract consumed by
``phoonnx.engines.mixertts.MixerTTSAdapter``: inputs ``token_ids``
[B, T] int64 + ``pace``/``pitch_mul``/``pitch_add`` [1] float32 +
``speaker``/``emotion`` [1] int32 -> output ``mel_spec`` [B, 80, T_mel].
"""
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from phoonnx_train.engines.base import TrainingEngineConfig
from phoonnx_train.engines.fastpitch import ForwardTTSTrainingEngine

if TYPE_CHECKING:  # heavy import — only needed for type annotations
    import pytorch_lightning as pl

_LOG = logging.getLogger(__name__)

# ONNX opset used for export (same floor as the FastPitch engine)
OPSET_VERSION = 14

# Quality tier -> Mixer-TTS hyper-param overrides. All encoder/decoder/
# aligner widths derive from symbols_embedding_dim; 384 is the paper
# configuration (~24M params), the smaller tiers shrink every Mixer block.
_QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "x-low": {"symbols_embedding_dim": 80},
    "medium": {"symbols_embedding_dim": 128},
    "high": {"symbols_embedding_dim": 384},
}


def resolve_module_kwargs(config: TrainingEngineConfig, **kwargs: Any) -> Dict[str, Any]:
    """Merge shared config + ``extra`` bag + call-site kwargs into the
    keyword arguments for ``MixerTTSModule``.

    Torch-free (pure dict handling) so it is unit-testable without the
    training stack. ``extra['quality']``, if present, expands to the
    matching preset (unknown names fall back to ``medium``); explicit
    ``extra``/kwargs keys win over preset values.
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
    merged.update(
        num_symbols=config.num_symbols,
        num_speakers=config.num_speakers,
        sample_rate=config.sample_rate,
    )
    return merged


class MixerTTSTrainingEngine(ForwardTTSTrainingEngine):
    """Training engine adapter for Mixer-TTS.

    Subclasses the FastPitch engine only to reuse its shared plumbing —
    the pyin-based F0 ``extra_preprocess`` sidecar cache and the tolerant
    ``load_checkpoint`` — the model/ONNX contract is Mixer-TTS's own.
    """

    def create_model(
        self,
        config: TrainingEngineConfig,
        dataset_paths: List[Path],
        **kwargs: Any,
    ) -> "pl.LightningModule":
        from phoonnx_train.mixertts.lightning import MixerTTSModule

        return MixerTTSModule(
            dataset=[str(p) for p in dataset_paths],
            **resolve_module_kwargs(config, **kwargs),
        )

    def export_onnx(
        self,
        checkpoint_path: Path,
        config_path: Path,
        output_dir: Path,
        **kwargs: Any,
    ) -> Path:
        """Export a Mixer-TTS checkpoint to ONNX.

        Contract matches ``phoonnx.engines.mixertts.MixerTTSAdapter``:
        ``token_ids`` [B,T] int64 + ``pace``/``pitch_mul``/``pitch_add``
        [1] float32 + ``speaker``/``emotion`` [1] int32 ->
        ``mel_spec`` [B, 80, T_mel]. The pitch controls operate in the
        normalized (z-scored) pitch domain, applied to the predicted
        pitch before the pitch embedding.
        """
        import json

        # validate the inputs before touching the heavy imports so a bad
        # path fails fast (and identically) with or without torch installed
        with open(config_path, "r", encoding="utf-8") as f:
            model_config: Dict[str, Any] = json.load(f)
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

        import torch

        from phoonnx_train.mixertts.lightning import MixerTTSModule
        from phoonnx_train.torch_compat import onnx_export_kwargs

        module: MixerTTSModule = MixerTTSModule.load_from_checkpoint(
            checkpoint_path, dataset=None, map_location="cpu",
        )
        model = module.model
        model.eval()

        num_symbols = module.hparams.num_symbols
        num_speakers = module.hparams.num_speakers

        class _InferWrapper(torch.nn.Module):
            """pace / pitch_mul / pitch_add / speaker / emotion are graph
            inputs so the MixerTTSAdapter's controls actually reach the
            model (traced as constants they would be silent no-ops).
            Single-speaker graphs simply never read speaker/emotion; the
            adapter filters its feed against the session inputs."""

            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, token_ids, pace, speaker, emotion,
                        pitch_mul, pitch_add):
                text_mask = (token_ids != self.m.padding_idx).unsqueeze(2)

                def pitch_transform(pitch, lens, mean, std):
                    return pitch * pitch_mul + pitch_add

                spect = self.m.infer(
                    text=token_ids, text_mask=text_mask, pace=pace,
                    speaker=speaker.long(), emotion=emotion.long(),
                    pitch_transform=pitch_transform,
                )
                return spect.transpose(1, 2).to(torch.float)  # [B, mel, T]

        wrapper = _InferWrapper(model)
        wrapper.eval()

        dummy_ids = torch.randint(low=1, high=max(2, num_symbols - 1),
                                  size=(1, 30), dtype=torch.long)
        # int32 to match the MixerTTSAdapter feed dtype; cast to long
        # inside the wrapper for the embedding lookup
        dummy_args = (
            dummy_ids,
            torch.ones(1),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            torch.ones(1),
            torch.zeros(1),
        )
        input_names = ["token_ids", "pace", "speaker", "emotion",
                       "pitch_mul", "pitch_add"]
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
                "model_type": "mixertts",
                "engine": "mixertts",
                "n_speakers": num_speakers,
                "n_vocab": num_symbols,
                "sample_rate": model_config.get("audio", {}).get(
                    "sample_rate", module.hparams.sample_rate),
                "mel_fmin": module.hparams.mel_fmin,
                "mel_fmax": module.hparams.mel_fmax,
                "mel_channels": module.hparams.mel_channels,
                "pitch_mean": module.hparams.get("pitch_mean", 0.0),
                "pitch_std": module.hparams.get("pitch_std", 1.0),
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

        _LOG.info("Exported Mixer-TTS ONNX model to %s", output_path)
        return output_path

    def quality_presets(self) -> Dict[str, Dict[str, Any]]:
        return _QUALITY_PRESETS
