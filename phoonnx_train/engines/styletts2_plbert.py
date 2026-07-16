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
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from phoonnx_train.engines.base import BaseTrainingEngine, TrainingEngineConfig
from phoonnx_train.engines.styletts2_aligner import _fused_kwargs
from phoonnx_train.styletts2.meldataset import TextCleaner

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


# ----------------------------------------------------------------------
# Dataset — upstream PL-BERT dataloader ported to the phonemize_corpus
# jsonl layout (pre-tokenized once; training reads ids only)
# ----------------------------------------------------------------------

class PLBertDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir: Path, cfg: PLBertConfig):
        data_dir = Path(data_dir)
        self.cfg = cfg
        self.cleaner = TextCleaner()
        self.mask_id = self.cleaner.word_index_dictionary["M"]  # upstream token_mask
        self.sep_phoneme = " "
        self.token_maps: Dict[str, int] = json.loads(
            (data_dir / "token_maps.json").read_text(encoding="utf-8"))
        self.rows = [json.loads(l) for l in
                     (data_dir / "data.jsonl").read_text(encoding="utf-8").splitlines()
                     if l.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def _word_mask_prob(self, word: str) -> float:
        cfg = self.cfg
        if not cfg.prosodic_masking:
            return cfg.word_mask_prob
        chars = set(word)
        if chars & _PROSODIC:
            return cfg.prosodic_mark_mask_prob
        if chars and chars <= _PUNCTUATION:
            return cfg.punct_mask_prob
        return cfg.word_mask_prob

    def __getitem__(self, idx: int):
        cfg = self.cfg
        row = self.rows[idx]
        phoneme_pool = "".join(row["phonemes"])

        phoneme = ""   # possibly-masked input
        labels = ""    # ground-truth phonemes
        words: List[int] = []
        masked_index: List[int] = []
        for word, ipa in zip(row["words"], row["phonemes"]):
            wid = self.token_maps.get(word, 1)  # <unk>
            words.extend([wid] * len(ipa))
            words.append(0)  # <sep>
            labels += ipa + self.sep_phoneme

            if random.random() < self._word_mask_prob(word):
                if random.random() < cfg.replace_prob:
                    if random.random() < (cfg.phoneme_mask_prob / cfg.replace_prob):
                        phoneme += "".join(random.choice(phoneme_pool)
                                           for _ in range(len(ipa)))
                    else:
                        phoneme += ipa
                else:
                    phoneme += "M" * len(ipa)
                masked_index.extend(range(len(phoneme) - len(ipa), len(phoneme)))
            else:
                phoneme += ipa
            phoneme += self.sep_phoneme

        if len(phoneme) > cfg.max_seq_length:
            start = random.randrange(0, len(phoneme) - cfg.max_seq_length)
            phoneme = phoneme[start:start + cfg.max_seq_length]
            labels = labels[start:start + cfg.max_seq_length]
            words = words[start:start + cfg.max_seq_length]
            masked_index = [m - start for m in masked_index
                            if start <= m < start + cfg.max_seq_length]

        phoneme_ids = self.cleaner(phoneme)
        label_ids = self.cleaner(labels)
        # any symbol dropped by the cleaner would desynchronize the three
        # streams (and masked_index) — fail fast like upstream
        assert len(phoneme_ids) == len(words), \
            f"phoneme/word streams desynchronized ({len(phoneme_ids)} vs {len(words)})"
        assert len(phoneme_ids) == len(label_ids), \
            f"phoneme/label streams desynchronized ({len(phoneme_ids)} vs {len(label_ids)})"
        return (torch.LongTensor(phoneme_ids),
                torch.LongTensor(words),
                torch.LongTensor(label_ids),
                torch.LongTensor(masked_index))


def _collate(batch):
    bsz = len(batch)
    max_len = max(b[0].size(0) for b in batch)
    phonemes = torch.zeros(bsz, max_len, dtype=torch.long)
    words = torch.zeros(bsz, max_len, dtype=torch.long)
    labels = torch.zeros(bsz, max_len, dtype=torch.long)
    lengths = torch.zeros(bsz, dtype=torch.long)
    masked = []
    for i, (p, w, l, m) in enumerate(batch):
        n = p.size(0)
        phonemes[i, :n] = p
        words[i, :n] = w
        labels[i, :n] = l
        lengths[i] = n
        masked.append(m)
    return phonemes, words, labels, lengths, masked


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

def _build_encoder(cfg: PLBertConfig):
    if cfg.backbone == "albert":
        from transformers import AlbertConfig, AlbertModel
        return AlbertModel(AlbertConfig(**cfg.model_params()))
    if cfg.backbone == "modernbert":
        from transformers import ModernBertModel

        from phoonnx_train.styletts2.Utils.PLBERT.util import \
            make_modernbert_config
        return ModernBertModel(make_modernbert_config(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_attention_heads,
            intermediate_size=cfg.intermediate_size,
            num_hidden_layers=cfg.num_hidden_layers,
            max_position_embeddings=cfg.max_position_embeddings))
    raise ValueError(f"Unknown PL-BERT backbone: {cfg.backbone!r}")


class MultiTaskPLBert(nn.Module):
    """Upstream ``MultiTaskModel``: encoder + MLM head + word (P2G) head."""

    def __init__(self, encoder: nn.Module, num_tokens: int, num_words: int,
                 hidden_size: int):
        super().__init__()
        self.encoder = encoder
        self.mask_predictor = nn.Linear(hidden_size, num_tokens)
        self.word_predictor = nn.Linear(hidden_size, num_words)

    def forward(self, phonemes, attention_mask=None):
        out = self.encoder(phonemes, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        return self.mask_predictor(hidden), self.word_predictor(hidden)


class PLBertModule(pl.LightningModule):

    def __init__(self, config: PLBertConfig, data_dir: Optional[Path] = None,
                 num_words: Optional[int] = None):
        super().__init__()
        self.config = config
        self.data_dir = Path(data_dir) if data_dir else None
        if num_words is None:
            if self.data_dir and (self.data_dir / "token_maps.json").is_file():
                token_maps = json.loads(
                    (self.data_dir / "token_maps.json").read_text(encoding="utf-8"))
                num_words = 1 + max(token_maps.values())
            else:
                raise ValueError("num_words or a data_dir with token_maps.json required")
        self.num_words = num_words
        self.model = MultiTaskPLBert(_build_encoder(config),
                                     num_tokens=config.vocab_size,
                                     num_words=num_words,
                                     hidden_size=config.hidden_size)
        if config.pretrained_dir:
            self._warm_start(config.pretrained_dir)
        if config.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        self.save_hyperparameters({"config": config.__dict__,
                                   "num_words": num_words})

    def _warm_start(self, plbert_dir: str) -> None:
        from phoonnx_train.styletts2.Utils.PLBERT.util import load_plbert
        bert = load_plbert(plbert_dir)
        enc_state = self.model.encoder.state_dict()
        filtered = {k: v for k, v in bert.state_dict().items()
                    if k in enc_state and v.shape == enc_state[k].shape}
        self.model.encoder.load_state_dict(filtered, strict=False)
        LOG.info("warm-started PL-BERT encoder from %s (%d/%d tensors)",
                 plbert_dir, len(filtered), len(enc_state))

    # ------------------------------------------------------------------
    @staticmethod
    def _length_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        ar = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return (ar < lengths.unsqueeze(1)).int()

    def _losses(self, batch):
        phonemes, words, labels, lengths, masked = batch
        attn = self._length_mask(lengths, phonemes.size(1))
        tokens_pred, words_pred = self.model(phonemes, attention_mask=attn)

        loss_vocab = 0.0
        for pred, target, n in zip(words_pred, words, lengths):
            loss_vocab += F.cross_entropy(pred[:n], target[:n])
        loss_vocab = loss_vocab / phonemes.size(0)

        loss_token = 0.0
        sizes = 1
        for pred, target, n, idx in zip(tokens_pred, labels, lengths, masked):
            if idx.numel() > 0:
                loss_token += F.cross_entropy(pred[:n][idx], target[:n][idx])
                sizes += 1
        loss_token = loss_token / sizes
        return loss_vocab, loss_token

    def training_step(self, batch, batch_idx):
        loss_vocab, loss_token = self._losses(batch)
        loss = loss_vocab + loss_token
        self.log_dict({"train/loss": loss, "train/vocab": loss_vocab,
                       "train/token": loss_token}, prog_bar=True)
        if (self.config.save_dir and self.global_step > 0
                and self.global_step % self.config.save_every_steps == 0):
            self.save_plbert_dir(Path(self.config.save_dir), self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss_vocab, loss_token = self._losses(batch)
        self.log("val_loss", loss_vocab + loss_token, prog_bar=True)

    def configure_optimizers(self):
        # upstream PL-BERT: plain AdamW(lr=1e-4), constant LR.
        # onecycle_scheduler adds a per-step OneCycle (10% warmup) on top.
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr,
                                **_fused_kwargs())
        if not self.config.onecycle_scheduler:
            return opt
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.config.lr,
            total_steps=max(int(self.trainer.estimated_stepping_batches), 2),
            pct_start=0.1)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    # ------------------------------------------------------------------
    def _dataloader(self, validation: bool):
        if not self.data_dir:
            return None
        ds = PLBertDataset(self.data_dir, self.config)
        kwargs: Dict[str, Any] = dict(
            batch_size=self.config.batch_size, collate_fn=_collate,
            shuffle=not validation, drop_last=not validation,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.config.num_workers > 0)
        if self.config.num_workers > 0:
            kwargs["prefetch_factor"] = 4
        return torch.utils.data.DataLoader(ds, **kwargs)

    def train_dataloader(self):
        return self._dataloader(validation=False)

    # ------------------------------------------------------------------
    def save_plbert_dir(self, out_dir: Path, step: Optional[int] = None) -> Path:
        """Write ``config.yml`` + ``step_N.t7`` consumable by ``load_plbert``
        (keys prefixed ``module.`` exactly like upstream's DDP save)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model = getattr(self.model, "_orig_mod", self.model)
        net = {f"module.{k}": v for k, v in model.state_dict().items()}
        step = step if step is not None else max(self.global_step, 1)
        ckpt = out_dir / f"step_{step}.t7"
        torch.save({"net": net, "step": step}, ckpt)
        (out_dir / "config.yml").write_text(yaml.safe_dump({
            "backbone": self.config.backbone,
            "model_params": self.config.model_params(),
        }))
        return ckpt

    def on_train_end(self):
        if self.config.save_dir:
            self.save_plbert_dir(Path(self.config.save_dir))


class PLBertTrainingEngine(BaseTrainingEngine):
    """Trains the prosodic text encoder consumed by ``--engine styletts2``."""

    def load_checkpoint(self, model: pl.LightningModule, checkpoint_path: Path,
                        **kwargs: Any) -> pl.LightningModule:
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
