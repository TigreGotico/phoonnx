"""PL-BERT training engine (``--engine styletts2-plbert``).

Lightning port of `yl4579/PL-BERT <https://github.com/yl4579/PL-BERT>`_ — the
phoneme-level masked language model StyleTTS2 stage-2 uses as prosodic text
encoder. Two backbones:

- ``albert`` (default) — upstream ``CustomAlbert``; checkpoints are
  byte-compatible with yl4579 ``load_plbert``.
- ``modernbert`` — same objectives on the ModernBERT architecture
  (needs ``transformers>=4.48``).

Dual heads per upstream: masked-phoneme MLM + phoneme-to-grapheme (word)
prediction. Optional ``prosodic_masking`` applies the proxectonos
inverse-frequency scheme (punctuation masked at 40%, ``!``/``?`` at 80%).

Dataset: a directory produced by
``python -m phoonnx_train.styletts2.phonemize_corpus plbert CORPUS OUT --lang xx``
(``data.jsonl`` + ``token_maps.json``).

Output (``save_plbert_dir``): ``config.yml`` + ``step_N.t7`` — the exact
layout ``load_plbert`` (and the StyleTTS2 engine's ``plbert_dir``) consumes.
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

_PUNCTUATION = set(';:,.¡¿—…"«»“” ')
_PROSODIC = set("!?")


@dataclass
class PLBertConfig:
    backbone: str = "albert"  # albert | modernbert
    vocab_size: int = 178
    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    max_position_embeddings: int = 512
    dropout: float = 0.1

    # masking (upstream defaults)
    word_mask_prob: float = 0.15
    phoneme_mask_prob: float = 0.1
    replace_prob: float = 0.2
    # proxectonos inverse-frequency scheme: mask punctuation words harder
    prosodic_masking: bool = False
    punct_mask_prob: float = 0.4
    prosodic_mark_mask_prob: float = 0.8

    lr: float = 1e-4
    onecycle_scheduler: bool = False  # upstream trains at constant LR
    batch_size: int = 32
    num_workers: int = 2
    max_seq_length: int = 512
    compile_model: bool = False

    save_dir: Optional[str] = None
    save_every_steps: int = 5000
    pretrained_dir: Optional[str] = None  # warm-start from an existing plbert_dir

    @classmethod
    def from_training_config(cls, cfg: TrainingEngineConfig) -> "PLBertConfig":
        extra = dict(cfg.extra)
        extra.pop("quality", None)
        extra.pop("validation_split", None)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: extra[k] for k in list(extra) if k in known})

    def model_params(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "max_position_embeddings": self.max_position_embeddings,
            "dropout": self.dropout,
        }


def __getattr__(name):
    """Lazily expose torch classes kept in the vendored package so this
    engine module imports torch-free."""
    if name in ("PLBertModule", "PLBertDataset", "MultiTaskPLBert", "_collate"):
        from phoonnx_train.styletts2 import plbert_module
        return getattr(plbert_module, name)
    raise AttributeError(name)


class PLBertTrainingEngine(BaseTrainingEngine):
    """Trains the prosodic text encoder consumed by ``--engine styletts2``."""

    def load_checkpoint(self, model: pl.LightningModule, checkpoint_path: Path,
                        **kwargs: Any) -> pl.LightningModule:
        import torch
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = ckpt.get("state_dict", ckpt.get("net", ckpt))
        stripped = {}
        for k, v in state.items():
            for prefix in ("model.", "module."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
            stripped[k] = v
        model.model.load_state_dict(stripped, strict=False)
        return model

    def create_model(self, config: TrainingEngineConfig,
                     dataset_paths: List[Path], **kwargs: Any) -> pl.LightningModule:
        from phoonnx_train.styletts2.plbert_module import PLBertModule
        pcfg = PLBertConfig.from_training_config(config)
        data_dir = None
        for p in dataset_paths:
            p = Path(p)
            if (p / "data.jsonl").is_file():
                data_dir = p
                break
        if data_dir is None:
            raise FileNotFoundError(
                "styletts2-plbert needs a dataset dir with data.jsonl + "
                "token_maps.json — build one with "
                "`python -m phoonnx_train.styletts2.phonemize_corpus plbert "
                f"CORPUS OUT --lang xx`; got: {[str(p) for p in dataset_paths]}")
        return PLBertModule(pcfg, data_dir=data_dir, **kwargs)

    def export_onnx(self, checkpoint_path: Path, config_path: Path,
                    output_dir: Path, **kwargs: Any) -> Path:
        raise NotImplementedError(
            "PL-BERT is a training-time auxiliary model; it is consumed as a "
            "checkpoint directory via plbert_dir.")

    def quality_presets(self) -> Dict[str, Dict[str, Any]]:
        return {
            "low": {"hidden_size": 256, "num_attention_heads": 4,
                    "num_hidden_layers": 4, "intermediate_size": 512},
            "medium": {},  # upstream 768x12
        }
