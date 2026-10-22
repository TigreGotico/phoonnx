"""LightningModule for MOSS-TTS-Nano supervised finetuning.

Objective (mirrors upstream ``finetuning/sft.py::compute_supervised_loss``):

1. run the global transformer over the packed rows -> ``global_hidden [B, S, H]``;
2. for every timestep, teacher-force the local transformer with
   ``[global_hidden, emb(text_target), emb(code_0), ..., emb(code_{n_vq-2})]``;
3. cross-entropy per head — the text head on position 0, audio head ``c`` on position
   ``c + 1`` — with ``ignore_index=-100``;
4. combine as a **weighted mean**: ``sum(w_i * CE_i) / sum(w_i)`` over the heads that
   have at least one live target. The default weights come from upstream's ``"1,32"``
   shorthand: 1 for the text head, 32 split evenly across the ``n_vq`` audio heads
   (2.0 each at ``n_vq=16``).

This is the global-local (RQ-Transformer) objective of arXiv:2603.18090, *not* the
delay-pattern one: there is no per-codebook time shift and no ``λ`` schedule over
codebook index — every audio head carries the same weight.

Optimizer and schedule defaults are upstream's: AdamW ``lr=1e-5``, ``betas=(0.9, 0.95)``,
``eps=1e-8``, ``weight_decay=0.1``, linear decay with a 3% warmup ratio, grad-norm clip 1.0.

Resume vs. finetune
-------------------
* **resume** — ``Trainer(...).fit(module, ckpt_path=...)``: Lightning restores model,
  optimizer, scheduler and global step exactly.
* **finetune** — build the module with ``warm_start_from=<upstream dir or .ckpt>``: only
  the weights are copied, and training starts a fresh optimizer/schedule.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from phoonnx_train.mosstts.config import MossTTSNanoConfig
from phoonnx_train.mosstts.dataset import IGNORE_INDEX
from phoonnx_train.mosstts.model import MossTTSNano

_LOGGER = logging.getLogger("mosstts.lightning")

SCHEDULER_CHOICES = (
    "linear",
    "cosine",
    "constant",
    "constant_with_warmup",
    "inverse_sqrt",
    "polynomial",
)


def parse_channelwise_loss_weight(spec: Union[str, Sequence[float]], n_heads: int) -> List[float]:
    """``"1,32"`` -> ``[1.0, 32/n_vq, ...]``; an explicit ``n_heads``-long list is kept as-is."""
    if isinstance(spec, str):
        values = [float(item.strip()) for item in spec.split(",") if item.strip()]
    else:
        values = [float(item) for item in spec]
    if len(values) == n_heads:
        resolved = values
    elif len(values) == 2 and n_heads > 1:
        text_weight, total_audio_weight = values
        resolved = [text_weight] + [total_audio_weight / float(n_heads - 1)] * (n_heads - 1)
    else:
        raise ValueError(
            f"channelwise_loss_weight expects either {n_heads} values or 2 values, got {len(values)}"
        )
    if any(weight < 0 for weight in resolved):
        raise ValueError("channelwise_loss_weight must not contain negative values")
    if sum(resolved) <= 0:
        raise ValueError("channelwise_loss_weight must sum to a positive value")
    return resolved


def build_lr_lambda(name: str, num_warmup_steps: int, num_training_steps: int, power: float = 1.0):
    """The subset of HuggingFace ``get_scheduler`` shapes upstream exposes, re-implemented."""
    if name not in SCHEDULER_CHOICES:
        raise ValueError(f"unsupported lr_scheduler_type={name!r}, expected one of {SCHEDULER_CHOICES}")
    num_warmup_steps = max(int(num_warmup_steps), 0)
    num_training_steps = max(int(num_training_steps), 1)

    def warmup(step: int) -> float:
        return float(step) / float(max(1, num_warmup_steps))

    def lr_lambda(step: int) -> float:
        if name == "constant":
            return 1.0
        if step < num_warmup_steps:
            return warmup(step)
        if name == "constant_with_warmup":
            return 1.0
        if name == "inverse_sqrt":
            return math.sqrt(max(1, num_warmup_steps) / float(max(step, 1)))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        if name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if name == "polynomial":
            return (1.0 - progress) ** power
        return 1.0 - progress  # linear

    return lr_lambda


class MossTTSNanoModule(pl.LightningModule):
    """Lightning wrapper around :class:`~phoonnx_train.mosstts.model.MossTTSNano`."""

    def __init__(
        self,
        config: Optional[Union[MossTTSNanoConfig, Dict[str, Any]]] = None,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.1,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.95,
        adam_eps: float = 1e-8,
        warmup_steps: int = 0,
        warmup_ratio: float = 0.03,
        lr_scheduler_type: str = "linear",
        max_train_steps: Optional[int] = None,
        channelwise_loss_weight: Union[str, Sequence[float]] = "1,32",
        warm_start_from: Optional[Union[str, Path]] = None,
        gradient_checkpointing: bool = False,
        attn_implementation: Optional[str] = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = MossTTSNanoConfig()
        elif isinstance(config, dict):
            config = MossTTSNanoConfig.from_dict(config)
        if attn_implementation is not None:
            config.attn_implementation = attn_implementation
            config.local_transformer_attn_implementation = attn_implementation
        self.save_hyperparameters(
            {
                "config": config.to_dict(),
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "adam_beta1": adam_beta1,
                "adam_beta2": adam_beta2,
                "adam_eps": adam_eps,
                "warmup_steps": warmup_steps,
                "warmup_ratio": warmup_ratio,
                "lr_scheduler_type": lr_scheduler_type,
                "max_train_steps": max_train_steps,
                "channelwise_loss_weight": channelwise_loss_weight,
                "gradient_checkpointing": gradient_checkpointing,
            }
        )
        self.config = config
        self.model = MossTTSNano(config)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(True)
        self.channelwise_loss_weight = parse_channelwise_loss_weight(
            channelwise_loss_weight, config.n_vq + 1
        )
        if warm_start_from is not None:
            from phoonnx_train.mosstts.warmstart import warm_start

            report = warm_start(self.model, warm_start_from)
            _LOGGER.info("warm start: %s", report.summary())

    # ------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        global_hidden, _ = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return global_hidden

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        global_hidden = self(input_ids, attention_mask)
        batch_size, seq_len, hidden_size = global_hidden.shape
        n_vq = self.config.n_vq

        flat_hidden = global_hidden.reshape(batch_size * seq_len, hidden_size)
        flat_labels = labels.reshape(batch_size * seq_len, n_vq + 1)

        # Timesteps whose labels are all -100 contribute nothing to any head, so the
        # local transformer never has to see them. Dropping them is exact, and turns the
        # local pass from O(padded length) into O(supervised length).
        live = (flat_labels != IGNORE_INDEX).any(dim=-1)
        if not bool(live.any()):
            raise RuntimeError(
                "every label is ignored — check the dataset packing and max_length"
            )
        flat_hidden = flat_hidden[live]
        flat_labels = flat_labels[live]

        text_logits, audio_logits = self.model.local_logits(flat_hidden, flat_labels)

        total_loss = torch.zeros((), device=flat_hidden.device, dtype=torch.float32)
        total_weight = 0.0
        per_head: Dict[str, torch.Tensor] = {}

        text_targets = flat_labels[:, 0]
        if bool((text_targets != IGNORE_INDEX).any()):
            text_loss = F.cross_entropy(text_logits.float(), text_targets, ignore_index=IGNORE_INDEX)
            weight = float(self.channelwise_loss_weight[0])
            total_loss = total_loss + weight * text_loss
            total_weight += weight
            per_head["loss_text"] = text_loss.detach()

        audio_targets = flat_labels[:, 1:]
        for channel_index in range(n_vq):
            channel_targets = audio_targets[:, channel_index]
            if not bool((channel_targets != IGNORE_INDEX).any()):
                continue
            channel_loss = F.cross_entropy(
                audio_logits[channel_index].float(), channel_targets, ignore_index=IGNORE_INDEX
            )
            weight = float(self.channelwise_loss_weight[channel_index + 1])
            total_loss = total_loss + weight * channel_loss
            total_weight += weight
            per_head[f"loss_vq{channel_index}"] = channel_loss.detach()

        if total_weight <= 0:
            raise RuntimeError("no head had a live target — check the dataset packing and max_length")
        return {"loss": total_loss / total_weight, "supervised_positions": int(live.sum()), **per_head}

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.compute_loss(batch)
        loss = outputs["loss"]
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        for name, value in outputs.items():
            if name.startswith("loss_"):
                self.log(f"train/{name}", value, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.compute_loss(batch)["loss"]
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    # ------------------------------------------------------------------
    def _resolve_total_steps(self) -> int:
        configured = self.hparams.get("max_train_steps")
        if configured:
            return int(configured)
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            estimated = getattr(trainer, "estimated_stepping_batches", None)
            if estimated is not None and math.isfinite(float(estimated)):
                return max(int(estimated), 1)
        return 1

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.hparams["learning_rate"]),
            weight_decay=float(self.hparams["weight_decay"]),
            betas=(float(self.hparams["adam_beta1"]), float(self.hparams["adam_beta2"])),
            eps=float(self.hparams["adam_eps"]),
        )
        total_steps = self._resolve_total_steps()
        warmup_steps = int(self.hparams["warmup_steps"])
        if warmup_steps <= 0 and float(self.hparams["warmup_ratio"]) > 0:
            warmup_steps = math.ceil(total_steps * float(self.hparams["warmup_ratio"]))
        scheduler = LambdaLR(
            optimizer,
            build_lr_lambda(str(self.hparams["lr_scheduler_type"]), warmup_steps, total_steps),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


__all__ = [
    "MossTTSNanoModule",
    "parse_channelwise_loss_weight",
    "build_lr_lambda",
    "SCHEDULER_CHOICES",
]
