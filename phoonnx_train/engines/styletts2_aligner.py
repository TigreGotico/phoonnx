"""StyleTTS2 text-aligner training engine (``--engine styletts2-aligner``).

Lightning port of `yl4579/AuxiliaryASR <https://github.com/yl4579/AuxiliaryASR>`_
— the mel→phoneme ASR whose attention provides the TMA alignment target in
StyleTTS2 stage-1. Train one per language, or warm-start from the English one
via ``pretrained_path`` for a fine-tune.

Loss = CTC(ctc_logits) + CrossEntropy(s2s decoder), exactly upstream
``trainer.py``. Dataset: ``train_list.txt``/``val_list.txt``
(``filename|phonemes|speaker``) + ``wavs/`` — same layout as the StyleTTS2
engine; phonemize raw text lists first with
``python -m phoonnx_train.styletts2.phonemize_corpus``.

Output (``save_asr_checkpoint``): ``epoch_XXXXX.pth`` + ``config.yml`` in the
exact layout ``load_ASR_models`` consumes — point the StyleTTS2 engine's
``asr_path``/``asr_config`` at them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from phoonnx_train.engines.base import BaseTrainingEngine, TrainingEngineConfig

if TYPE_CHECKING:  # heavy imports — only for type annotations
    import pytorch_lightning as pl

LOG = logging.getLogger(__name__)


def _fused_kwargs() -> Dict[str, Any]:
    """``fused=True`` for AdamW needs CUDA and torch>=2."""
    import torch
    if torch.cuda.is_available() and int(torch.__version__.split(".")[0]) >= 2:
        return {"fused": True}
    return {}


def load_aux_state_dict(checkpoint_path) -> Dict[str, Any]:
    """Load an aux-model checkpoint in any of its layouts: a Lightning
    ``state_dict`` (keys ``model.*``), the aux-engine ``{"model": ...}`` /
    ``{"net": ...}`` layouts, or a bare state_dict — returned with the
    Lightning ``model.`` prefix stripped so it loads into the inner module."""
    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model", "net"):
        if isinstance(ckpt.get(key), dict):
            ckpt = ckpt[key]
            break
    return {(k[len("model."):] if k.startswith("model.") else k): v
            for k, v in ckpt.items()}


@dataclass
class AlignerConfig:
    sample_rate: int = 24000
    n_mels: int = 80
    n_token: int = 178
    hidden_dim: int = 256
    token_embedding_dim: int = 512
    n_layers: int = 6

    lr: float = 5e-4
    batch_size: int = 32
    num_workers: int = 2
    grad_clip_value: float = 5.0  # upstream clip_grad_value_(…, 5)
    compile_model: bool = False

    root_path: str = ""
    save_dir: Optional[str] = None  # where epoch_XXXXX.pth + config.yml land
    pretrained_path: Optional[str] = None  # warm-start (e.g. English aligner)

    @classmethod
    def from_training_config(cls, cfg: TrainingEngineConfig) -> "AlignerConfig":
        extra = dict(cfg.extra)
        extra.pop("quality", None)
        extra.pop("validation_split", None)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: extra[k] for k in list(extra) if k in known}
        kwargs.setdefault("n_token", cfg.num_symbols)
        kwargs.setdefault("sample_rate", cfg.sample_rate)
        return cls(**kwargs)


def __getattr__(name):
    """Lazily expose the torch ``LightningModule`` (kept in the vendored
    package so this engine module imports torch-free)."""
    if name == "AlignerModule":
        from phoonnx_train.styletts2.aligner_module import AlignerModule
        return AlignerModule
    raise AttributeError(name)


def _read_lists(dataset_paths: List[Path]):
    train_list: List[str] = []
    val_list: List[str] = []
    root = ""
    for p in dataset_paths:
        p = Path(p)
        if p.is_dir():
            tl, vl = p / "train_list.txt", p / "val_list.txt"
            if tl.is_file():
                train_list.extend(tl.read_text(encoding="utf-8").splitlines())
            if vl.is_file():
                val_list.extend(vl.read_text(encoding="utf-8").splitlines())
            if (p / "wavs").is_dir():
                root = str(p / "wavs")
            elif not root:
                root = str(p)
        elif p.is_file():
            lines = p.read_text(encoding="utf-8").splitlines()
            (val_list if "val" in p.stem else train_list).extend(lines)
            if not root:
                root = str(p.parent)
    return train_list, val_list, root


class AlignerTrainingEngine(BaseTrainingEngine):
    """Trains the AuxiliaryASR text aligner consumed by ``--engine styletts2``."""

    def load_checkpoint(self, model: pl.LightningModule, checkpoint_path: Path,
                        **kwargs: Any) -> pl.LightningModule:
        state = load_aux_state_dict(checkpoint_path)
        model.model.load_state_dict(state, strict=False)
        return model

    def create_model(self, config: TrainingEngineConfig,
                     dataset_paths: List[Path], **kwargs: Any) -> pl.LightningModule:
        from phoonnx_train.styletts2.aligner_module import AlignerModule
        acfg = AlignerConfig.from_training_config(config)
        train_list, val_list, root = _read_lists(dataset_paths)
        if not train_list:
            raise FileNotFoundError(
                "styletts2-aligner needs a train_list.txt "
                "(filename|phonemes|speaker per line) in the dataset dir(s): "
                f"{[str(p) for p in dataset_paths]}")
        if not acfg.root_path:
            acfg.root_path = root
        return AlignerModule(acfg, train_list=train_list, val_list=val_list, **kwargs)

    def export_onnx(self, checkpoint_path: Path, config_path: Path,
                    output_dir: Path, **kwargs: Any) -> Path:
        raise NotImplementedError(
            "The text aligner is a training-time auxiliary model; it is "
            "consumed as a torch checkpoint via asr_path/asr_config.")

    def quality_presets(self) -> Dict[str, Dict[str, Any]]:
        return {
            "low": {"hidden_dim": 128, "token_embedding_dim": 256, "n_layers": 3},
            "medium": {},  # upstream AuxiliaryASR defaults
        }
