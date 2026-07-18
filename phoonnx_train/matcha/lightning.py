"""Matcha-TTS LightningModule.

Model code follows https://github.com/shivammehta25/Matcha-TTS (MIT),
adapted to pytorch_lightning and the shared phoonnx dataset format:
the module owns its dataloaders (``Trainer.fit(model)`` with no
datamodule) and uses a plain Adam optimizer.
"""
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from phoonnx_train.matcha.dataset import MatchaDataset, collate_matcha
from phoonnx_train.matcha.flow_matching import CFM
from phoonnx_train.matcha.text_encoder import TextEncoder
from phoonnx_train.matcha.utils import (
    denormalize,
    duration_loss,
    fix_len_compatibility,
    generate_path,
    sequence_mask,
)
from phoonnx_train.matcha.mas import maximum_path


class MatchaTTS(pl.LightningModule):  # 🍵
    def __init__(
        self,
        n_vocab: int,
        n_spks: int,
        spk_emb_dim: int,
        n_feats: int,
        encoder: Any,
        decoder: Any,
        cfm: Any,
        data_statistics: Dict[str, float],
        out_size: Optional[int] = None,
        prior_loss: bool = True,
        use_precomputed_durations: bool = False,
        dataset: Optional[List[str]] = None,
        sample_rate: int = 22050,
        batch_size: int = 16,
        num_workers: int = 1,
        validation_split: float = 0.05,
        learning_rate: float = 1e-4,
        max_phoneme_ids: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_feats = n_feats
        self.out_size = out_size
        self.prior_loss = prior_loss
        self.use_precomputed_durations = use_precomputed_durations

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)

        self.encoder = TextEncoder(
            encoder.encoder_type,
            encoder.encoder_params,
            encoder.duration_predictor_params,
            n_vocab,
            n_spks,
            spk_emb_dim,
        )

        self.decoder = CFM(
            in_channels=2 * encoder.encoder_params.n_feats,
            out_channel=encoder.encoder_params.n_feats,
            cfm_params=cfm,
            decoder_params=decoder,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        self.register_buffer("mel_mean", torch.tensor(data_statistics["mel_mean"]))
        self.register_buffer("mel_std", torch.tensor(data_statistics["mel_std"]))

        self._train_dataset = None
        self._val_dataset = None

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _load_datasets(self) -> None:
        if self._train_dataset is not None or not self.hparams.dataset:
            return
        full = MatchaDataset(
            [Path(p) for p in self.hparams.dataset],
            n_feats=self.n_feats,
            sample_rate=self.hparams.sample_rate,
            mel_mean=float(self.mel_mean),
            mel_std=float(self.mel_std),
            max_phoneme_ids=self.hparams.max_phoneme_ids,
        )
        val_size = min(int(len(full) * self.hparams.validation_split), len(full) - 1)
        self._train_dataset, self._val_dataset = random_split(
            full, [len(full) - val_size, val_size]
        )

    def train_dataloader(self):
        self._load_datasets()
        return DataLoader(
            self._train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_matcha,
        )

    def val_dataloader(self):
        self._load_datasets()
        return DataLoader(
            self._val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_matcha,
        )

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def get_losses(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        dur_loss, prior_loss, diff_loss, _ = self(
            x=batch["x"],
            x_lengths=batch["x_lengths"],
            y=batch["y"],
            y_lengths=batch["y_lengths"],
            spks=batch.get("spks"),
            out_size=self.out_size,
            durations=batch.get("durations"),
        )
        return {"dur_loss": dur_loss, "prior_loss": prior_loss, "diff_loss": diff_loss}

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        loss_dict = self.get_losses(batch)
        for name, loss in loss_dict.items():
            self.log(f"sub_loss/train_{name}", loss, on_step=True, on_epoch=True, sync_dist=True)
        total_loss = sum(loss_dict.values())
        self.log("loss/train", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        loss_dict = self.get_losses(batch)
        for name, loss in loss_dict.items():
            self.log(f"sub_loss/val_{name}", loss, on_epoch=True, sync_dist=True)
        total_loss = sum(loss_dict.values())
        self.log("loss/val", total_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return total_loss

    # ------------------------------------------------------------------
    # Model (follows upstream matcha/models/matcha_tts.py)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def synthesise(self, x, x_lengths, n_timesteps, temperature=1.0, spks=None, length_scale=1.0):
        """Generate a mel-spectrogram from phoneme ids."""
        if self.n_spks > 1:
            spks = self.spk_emb(spks.long())

        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()
        y_max_length_ = fix_len_compatibility(y_max_length)

        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        decoder_outputs = self.decoder(mu_y, y_mask, n_timesteps, temperature, spks)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return {
            "encoder_outputs": encoder_outputs,
            "decoder_outputs": decoder_outputs,
            "attn": attn[:, :, :y_max_length],
            "mel": denormalize(decoder_outputs, float(self.mel_mean), float(self.mel_std)),
            "mel_lengths": y_lengths,
        }

    def forward(self, x, x_lengths, y, y_lengths, spks=None, out_size=None, cond=None, durations=None):
        """Duration, prior and flow-matching losses (MAS alignment)."""
        if self.n_spks > 1:
            spks = self.spk_emb(spks)

        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        if self.use_precomputed_durations:
            attn = generate_path(durations.squeeze(1), attn_mask.squeeze(1))
        else:
            with torch.no_grad():
                const = -0.5 * math.log(2 * math.pi) * self.n_feats
                factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
                y_square = torch.matmul(factor.transpose(1, 2), y**2)
                y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
                mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
                log_prior = y_square - y_mu_double + mu_square + const

                attn = maximum_path(log_prior, attn_mask.squeeze(1))
                attn = attn.detach()  # b, t_text, T_mel

        # Duration loss between predicted log-durations and MAS durations
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Cut a mel segment to bound decoder memory (Grad-TTS hack)
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor(
                [torch.tensor(random.choice(range(start, end)) if end > start else 0) for start, end in offset_ranges]
            ).to(y_lengths)
            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)

            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]

            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask

        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        diff_loss, _ = self.decoder.compute_loss(x1=y, mask=y_mask, mu=mu_y, spks=spks, cond=cond)

        if self.prior_loss:
            prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        else:
            prior_loss = torch.tensor(0.0, device=y.device)

        return dur_loss, prior_loss, diff_loss, attn
