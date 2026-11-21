"""Rápido LightningModule: flow matching over AudioVAE latents.

Upstream Rápido ships no training code; this loop is our own. Per batch:

1. The text encoder turns letter ids into features ``h``. A linear prior
   head projects ``h`` into the 64-dim latent space.
2. Monotonic alignment search over the normalized latents and the prior
   means gives each letter a duration in frames (see ``alignment.py``).
   Nothing flows back through the search, as in Matcha-TTS.
3. The prior loss pulls the prior means toward the latents they align to.
   The duration predictor regresses the log durations on detached ``h``.
4. The aligned durations give the word and frame layout. The aligner turns
   ``h`` and that layout into a per-frame condition, which is zeroed for a
   ``cond_drop`` share of the batch so the model also learns the
   unconditional field that classifier-free guidance needs.
5. The DiT regresses the straight-line velocity ``z - x0`` at a random time
   ``t`` between noise ``x0`` (t = 0) and the latents ``z`` (t = 1).
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from phoonnx_train.rapido.alignment import align, batch_layout
from phoonnx_train.rapido.dataset import (
    LATENT_DIM,
    RapidoDataset,
    collate_rapido,
    ensure_latents,
    latent_cache_path,
    load_or_compute_statistics,
    load_utterances,
    strip_blanks,
)
from phoonnx_train.rapido.model import RapidoConfig, RapidoModel

_LOGGER = logging.getLogger(__name__)

# parameters of the text side; everything else in ``net`` is the DiT backbone
TEXT_PREFIXES = ("text.", "chardur.")
CONDITION_PREFIXES = TEXT_PREFIXES + ("aligner.",)


class RapidoModule(pl.LightningModule):
    def __init__(
        self,
        num_symbols: int,
        num_speakers: int,
        boundary_ids: Sequence[int],
        blank: Optional[int],
        sample_rate: int,
        phoneme_id_map: Optional[Dict[str, int]] = None,
        dataset: Optional[List[str]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        batch_size: int = 16,
        num_workers: int = 1,
        validation_split: float = 0.05,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        cond_drop: float = 0.1,
        mas_upsample: int = 4,
        prior_weight: float = 1.0,
        duration_weight: float = 1.0,
        freeze_backbone_steps: int = 0,
        text_only: bool = False,
        max_phoneme_ids: Optional[int] = None,
        latent_device: str = "cpu",
        **_ignored: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["_ignored"])
        config = RapidoConfig(**(model_params or {}))
        self.model = RapidoModel(config, n_vocab=num_symbols, n_speakers=num_speakers)
        self.prior = torch.nn.Linear(config.d, LATENT_DIM)
        self.boundary_ids = frozenset(boundary_ids)
        self._train_dataset = None
        self._val_dataset = None

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        if self._train_dataset is not None or not self.hparams.dataset:
            return
        paths = [Path(p) for p in self.hparams.dataset]
        utterances = load_utterances(paths, self.hparams.max_phoneme_ids)
        ensure_latents(utterances, self.hparams.sample_rate, self.hparams.latent_device)
        kept = [u for u in utterances if self._alignable(u)]
        if len(kept) < len(utterances):
            _LOGGER.warning("skipped %d utterances with more letters than %d x frames",
                            len(utterances) - len(kept), self.hparams.mas_upsample)
        mean, std = load_or_compute_statistics(paths, kept)
        self.model.lat_mean.copy_(torch.tensor(mean))
        self.model.lat_std.copy_(torch.tensor(std))
        full = RapidoDataset(kept, self.hparams.blank)
        val_size = min(int(len(full) * self.hparams.validation_split), len(full) - 1)
        self._train_dataset, self._val_dataset = random_split(
            full, [len(full) - val_size, val_size],
            generator=torch.Generator().manual_seed(1234))

    def _alignable(self, utt: Dict[str, Any]) -> bool:
        frames = torch.load(latent_cache_path(utt["audio_norm_path"])).shape[0]
        letters = len(strip_blanks(utt["phoneme_ids"], self.hparams.blank))
        return letters <= frames * self.hparams.mas_upsample

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=shuffle,
                          num_workers=self.hparams.num_workers, collate_fn=collate_rapido)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self._val_dataset, shuffle=False)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents - self.model.lat_mean) / self.model.lat_std

    def forward(self, batch: Dict[str, torch.Tensor], cond_drop: float = 0.0,
                t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Flow, prior and duration losses of a collated batch.

        Also returns the aligned letter durations (``durations``) and the
        word layout (``cw``) for inspection.
        """
        ids, mask, fmask = batch["ids"], batch["mask"], batch["fmask"]
        z = self.normalize(batch["latents"])
        letters = mask.float()
        frames = fmask.float()[..., None]

        h = self.model.net.text(ids, mask)
        mu = self.prior(h)
        attn = align(mu.detach(), z, mask, fmask, self.hparams.mas_upsample)
        durations = attn.sum(-1)

        mu_y = attn.transpose(1, 2) @ mu
        prior_loss = (((z - mu_y) ** 2) * frames).sum() / (frames.sum() * LATENT_DIM)

        log_dur = self.model.net.chardur.log_duration(h.detach(), mask)
        dur_loss = (((log_dur - torch.log1p(durations)) ** 2) * letters).sum() / letters.sum()

        cw, wstart, fw, fp = batch_layout(ids, mask, durations, fmask, self.boundary_ids)
        cond = self.model.condition(h, durations, mask, cw, wstart, fw, fp, batch["speakers"])
        if cond_drop > 0:
            keep = torch.rand(cond.shape[0], 1, 1, device=cond.device) >= cond_drop
            cond = cond * keep

        x0 = torch.randn_like(z)
        if t is None:
            t = torch.rand(z.shape[0], device=z.device)
        tt = t[:, None, None]
        xt = (1 - tt) * x0 + tt * z
        v = self.model.velocity(xt, cond, t, fmask)
        flow_loss = (((v - (z - x0)) ** 2) * frames).sum() / (frames.sum() * LATENT_DIM)

        return {"flow_loss": flow_loss, "prior_loss": prior_loss, "dur_loss": dur_loss,
                "durations": durations, "cw": cw}

    def _total(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        return (losses["flow_loss"]
                + self.hparams.prior_weight * losses["prior_loss"]
                + self.hparams.duration_weight * losses["dur_loss"])

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        losses = self(batch, cond_drop=self.hparams.cond_drop)
        total = self._total(losses)
        size = batch["ids"].shape[0]
        for name in ("flow_loss", "prior_loss", "dur_loss"):
            self.log(f"train/{name}", losses[name], on_step=True, on_epoch=True, batch_size=size)
        self.log("loss/train", total, on_step=True, on_epoch=True, prog_bar=True, batch_size=size)
        return total

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        b = batch["ids"].shape[0]
        # a fixed time grid keeps validation numbers comparable across epochs
        t = (torch.arange(b, device=self.device) + 0.5) / b
        losses = self(batch, t=t)
        total = self._total(losses)
        for name in ("flow_loss", "prior_loss", "dur_loss"):
            self.log(f"val/{name}", losses[name], on_epoch=True, batch_size=b)
        self.log("loss/val", total, on_epoch=True, prog_bar=True, batch_size=b)
        return total

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def _trainable(self, name: str) -> bool:
        if name.startswith("prior."):
            return True
        if not name.startswith("model.net."):
            return not self.hparams.text_only
        local = name[len("model.net."):]
        if self.hparams.text_only:
            return local.startswith(TEXT_PREFIXES)
        if self.global_step < self.hparams.freeze_backbone_steps:
            return local.startswith(CONDITION_PREFIXES)
        return True

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        for name, param in self.named_parameters():
            param.requires_grad_(self._trainable(name))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate,
                                 weight_decay=self.hparams.weight_decay)
