"""StyleTTS2 pitch-extractor training engine (``--engine styletts2-pitch``).

Lightning port of `yl4579/PitchExtractor <https://github.com/yl4579/PitchExtractor>`_
— the JDCNet F0 estimator StyleTTS2 stage-1 uses for ground-truth pitch.
F0 estimation is largely language-independent, so the English checkpoint
usually transfers and training one is optional; ``pretrained_path``
warm-starts a fine-tune.

Losses per upstream: ``lambda_f0 * SmoothL1(f0)`` + ``BCEWithLogits`` on the
voicing/silence labels, with pyworld harvest (dio fallback) ground-truth F0
cached next to the audio. Dataset: same ``train_list.txt``/``wavs/`` layout
as the other StyleTTS2 engines (the text field is unused).

Output (``save_f0_checkpoint``): a ``.t7`` with ``{"net": state_dict}`` —
exactly what ``load_F0_models`` consumes via the StyleTTS2 engine's
``f0_path``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from phoonnx_train.engines.base import BaseTrainingEngine, TrainingEngineConfig
from phoonnx_train.engines.styletts2_aligner import (_read_lists,
                                                     load_aux_state_dict)

if TYPE_CHECKING:  # heavy imports — only for type annotations
    import pytorch_lightning as pl

LOG = logging.getLogger(__name__)


@dataclass
class PitchConfig:
    sample_rate: int = 24000
    n_mels: int = 80
    seq_len: int = 192  # JDCNet training segment (frames)

    lr: float = 3e-4  # upstream optimizer_params.lr
    lambda_f0: float = 0.1  # upstream loss_params.lambda_f0
    batch_size: int = 32
    num_workers: int = 2
    compile_model: bool = False

    root_path: str = ""
    save_dir: Optional[str] = None
    pretrained_path: Optional[str] = None  # warm-start (e.g. English bst.t7)

    @classmethod
    def from_training_config(cls, cfg: TrainingEngineConfig) -> "PitchConfig":
        extra = dict(cfg.extra)
        extra.pop("quality", None)
        extra.pop("validation_split", None)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: extra[k] for k in list(extra) if k in known}
        kwargs.setdefault("sample_rate", cfg.sample_rate)
        return cls(**kwargs)


def __getattr__(name):
    """Lazily expose torch classes kept in the vendored package so this
    engine module imports torch-free."""
    if name in ("PitchModule", "PitchSegmentDataset"):
        from phoonnx_train.styletts2 import pitch_module
        return getattr(pitch_module, name)
    raise AttributeError(name)


class PitchTrainingEngine(BaseTrainingEngine):
    """Trains the JDC pitch extractor consumed by ``--engine styletts2``."""

    def load_checkpoint(self, model: pl.LightningModule, checkpoint_path: Path,
                        **kwargs: Any) -> pl.LightningModule:
        state = load_aux_state_dict(checkpoint_path)
        model.model.load_state_dict(state, strict=False)
        return model

    def create_model(self, config: TrainingEngineConfig,
                     dataset_paths: List[Path], **kwargs: Any) -> pl.LightningModule:
        from phoonnx_train.styletts2.pitch_module import PitchModule
        pcfg = PitchConfig.from_training_config(config)
        train_list, val_list, root = _read_lists(dataset_paths)
        if not train_list:
            raise FileNotFoundError(
                "styletts2-pitch needs a train_list.txt (filename|text|speaker "
                f"per line) in the dataset dir(s): {[str(p) for p in dataset_paths]}")
        if not pcfg.root_path:
            pcfg.root_path = root
        return PitchModule(pcfg, train_list=train_list, val_list=val_list, **kwargs)

    def export_onnx(self, checkpoint_path: Path, config_path: Path,
                    output_dir: Path, **kwargs: Any) -> Path:
        raise NotImplementedError(
            "The pitch extractor is a training-time auxiliary model; it is "
            "consumed as a torch checkpoint via f0_path.")

    def quality_presets(self) -> Dict[str, Dict[str, Any]]:
        return {"medium": {}}  # JDCNet has a fixed upstream architecture
