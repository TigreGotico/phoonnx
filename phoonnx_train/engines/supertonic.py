"""SuperTonic training engine (``--engine supertonic``).

SuperTonic (Kim et al., Supertone Inc., "SupertonicTTS", arXiv:2503.23108) is a
three-stage TTS: a GAN speech autoencoder, a flow-matching text-to-latent module,
and an utterance-level duration predictor. Each stage trains independently, so
this engine selects a stage through ``extra["stage"]`` (``autoencoder`` /
``text_to_latent`` / ``duration_predictor``) and builds the matching
LightningModule from :mod:`phoonnx_train.supertonic.lightning`.

The model, losses and export live in the vendored ``phoonnx_train/supertonic/``
package. ``export_onnx`` produces the four-graph contract the phoonnx
``supertonic`` inference engine consumes (``duration_predictor.onnx``,
``text_encoder.onnx``, ``vector_estimator.onnx``, ``vocoder.onnx`` plus
``tts.json`` and ``unicode_indexer.json``); it needs all three stage checkpoints,
passed as ``extra["autoencoder_ckpt"] / ["text_to_latent_ckpt"] /
["duration_predictor_ckpt"]``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from phoonnx_train.engines.base import BaseTrainingEngine, TrainingEngineConfig

if TYPE_CHECKING:
    import pytorch_lightning as pl

LOG = logging.getLogger(__name__)

_STAGES = {
    "autoencoder": "AutoencoderModule",
    "text_to_latent": "TextToLatentModule",
    "duration_predictor": "DurationPredictorModule",
}

# model-size tiers; "base" keeps the package defaults, "low" is a tiny CI tier
_QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "base": {},
    "low": {"quality_low": True},
}


class SuperTonicTrainingEngine(BaseTrainingEngine):
    """Three-stage SuperTonic training engine."""

    def _config_for(self, config: TrainingEngineConfig):
        from phoonnx_train.supertonic.config import tiny_config, SuperTonicConfig
        if config.extra.get("quality") == "low" or config.extra.get("quality_low"):
            cfg = tiny_config(vocab_size=config.num_symbols)
        else:
            cfg = SuperTonicConfig(vocab_size=config.num_symbols)
        cfg.ae.sample_rate = config.sample_rate
        return cfg

    def create_model(self, config: TrainingEngineConfig, dataset_paths: List[Path],
                     **kwargs: Any) -> "pl.LightningModule":
        import phoonnx_train.supertonic.lightning as lm

        extra = dict(config.extra)
        stage = extra.pop("stage", "autoencoder")
        if stage not in _STAGES:
            raise KeyError(f"unknown supertonic stage {stage!r}; expected {list(_STAGES)}")
        extra.pop("quality", None)
        extra.pop("quality_low", None)
        cfg = self._config_for(config)
        module_cls = getattr(lm, _STAGES[stage])
        return module_cls(config=cfg, dataset=[str(p) for p in dataset_paths], **extra, **kwargs)

    def export_onnx(self, checkpoint_path: Path, config_path: Path,
                    output_dir: Path, **kwargs: Any) -> Path:
        from phoonnx_train.supertonic.export_onnx import export_from_checkpoints

        paths = export_from_checkpoints(
            str(output_dir),
            autoencoder_ckpt=kwargs.get("autoencoder_ckpt", str(checkpoint_path)),
            text_to_latent_ckpt=kwargs["text_to_latent_ckpt"],
            duration_predictor_ckpt=kwargs["duration_predictor_ckpt"])
        return paths["vector_estimator"]

    def quality_presets(self) -> Dict[str, Dict[str, Any]]:
        return _QUALITY_PRESETS
