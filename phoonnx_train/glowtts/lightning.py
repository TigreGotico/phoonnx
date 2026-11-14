"""
GlowTTS LightningModule.

Reuses phoonnx_train's existing dataset / collation / mel-extraction
pipeline (``phoonnx_train.vits.dataset.PhoonnxDataset`` +
``UtteranceCollate`` + ``mel_processing.spec_to_mel_torch``) unmodified.
GlowTTS trains on mel spectrograms only (no waveform / vocoder training) —
the linear spectrogram already produced by that pipeline is converted to mel
on the fly, exactly like VITS's own mel-domain loss term does.

Loss (reconstructed from the GlowTTS paper §3.2):

  - ``loss_mle``: exact negative log-likelihood of the target mel under the
    flow-transformed Gaussian prior (change-of-variables: NLL(z) - logdet).
  - ``loss_dur``: MSE between predicted and MAS-derived log-durations.
"""
import logging
from pathlib import Path
from typing import List, Optional, Union

import pytorch_lightning as pl
import math

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from phoonnx_train.vits.dataset import Batch, PhoonnxDataset, UtteranceCollate
from phoonnx_train.vits.mel_processing import spec_to_mel_torch

from phoonnx_train.glowtts.glow import GlowTTSGenerator

_LOGGER = logging.getLogger("glowtts.lightning")


class GlowTTSModel(pl.LightningModule):
    def __init__(
        self,
        num_symbols: int,
        num_speakers: int = 1,
        # audio / mel
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        mel_channels: int = 80,
        sample_rate: int = 22050,
        mel_fmin: float = 0.0,
        # pinned to 8 kHz — the HiFi-GAN-family vocoder configs this mel is
        # consumed by train with fmax=8000; leaving it None (= Nyquist) makes
        # the acoustic model and vocoder disagree on the mel basis. Recorded
        # in the exported ONNX metadata (see engines/glowtts.py) so the
        # vocoder pairing can be validated downstream.
        mel_fmax: Optional[float] = 8000.0,
        # model
        hidden_channels: int = 192,
        filter_channels: int = 768,
        filter_channels_dp: int = 256,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        prenet_n_layers: int = 3,
        dec_hidden_channels: int = 192,
        dec_kernel_size: int = 5,
        dec_dilation_rate: int = 1,
        dec_n_blocks: int = 12,
        dec_n_layers: int = 4,
        n_sqz: int = 2,
        gin_channels: int = 512,
        # training
        dataset: Optional[List[Union[str, Path]]] = None,
        learning_rate: float = 1e-3,
        betas=(0.9, 0.98),
        eps: float = 1e-9,
        warmup_steps: int = 4000,  # Noam warmup, as in the reference recipe
        batch_size: int = 1,
        c_dur: float = 1.0,
        num_workers: int = 1,
        seed: int = 1234,
        num_test_examples: int = 5,
        validation_split: float = 0.1,
        max_phoneme_ids: Optional[int] = None,
        segment_size: int = 8192,  # kept for collate-fn API parity with VITS, unused for audio slicing
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        eff_gin = gin_channels if num_speakers > 1 else 0
        self.model_g = GlowTTSGenerator(
            n_vocab=num_symbols,
            n_mels=mel_channels,
            n_speakers=num_speakers,
            gin_channels=eff_gin,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            filter_channels_dp=filter_channels_dp,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            prenet_n_layers=prenet_n_layers,
            dec_hidden_channels=dec_hidden_channels,
            dec_kernel_size=dec_kernel_size,
            dec_dilation_rate=dec_dilation_rate,
            dec_n_blocks=dec_n_blocks,
            dec_n_layers=dec_n_layers,
            n_sqz=n_sqz,
        )

        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
        self._load_datasets(validation_split, num_test_examples, max_phoneme_ids)

    # ------------------------------------------------------------------
    # Dataset wiring (shared pipeline)
    # ------------------------------------------------------------------

    def _load_datasets(
        self,
        validation_split: float,
        num_test_examples: int,
        max_phoneme_ids: Optional[int] = None,
    ):
        if not self.hparams.dataset:
            _LOGGER.debug("No dataset to load")
            return

        # train.py passes the preprocessed dataset *directory*; resolve to the
        # dataset.jsonl inside it (same convention as the mixer/zipvoice engines)
        paths = [
            (Path(p) / "dataset.jsonl") if Path(p).is_dir() else Path(p)
            for p in self.hparams.dataset
        ]
        full_dataset = PhoonnxDataset(paths, max_phoneme_ids=max_phoneme_ids)
        valid_set_size = int(len(full_dataset) * validation_split)
        train_set_size = len(full_dataset) - valid_set_size - num_test_examples

        self._train_dataset, self._test_dataset, self._val_dataset = random_split(
            full_dataset, [train_set_size, num_test_examples, valid_set_size]
        )

    def _collate(self) -> UtteranceCollate:
        return UtteranceCollate(
            is_multispeaker=self.hparams.num_speakers > 1,
            segment_size=self.hparams.segment_size,
        )

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset, collate_fn=self._collate(),
            num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset, collate_fn=self._collate(),
            num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset, collate_fn=self._collate(),
            num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size,
        )

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------

    def forward(self, text, text_lengths, scales, sid=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        mel, _mel_lengths = self.model_g.infer(
            text, text_lengths, noise_scale=noise_scale, length_scale=length_scale, sid=sid,
        )
        return mel

    def _mel_from_batch(self, batch: Batch) -> torch.Tensor:
        return spec_to_mel_torch(
            batch.spectrograms,
            self.hparams.filter_length,
            self.hparams.mel_channels,
            self.hparams.sample_rate,
            self.hparams.mel_fmin,
            self.hparams.mel_fmax,
        )

    def _compute_loss(self, batch: Batch):
        x, x_lengths = batch.phoneme_ids, batch.phoneme_lengths
        mel = self._mel_from_batch(batch)
        mel_lengths = batch.spectrogram_lengths
        sid = batch.speaker_ids if batch.speaker_ids is not None else None

        z, logdet, m_p, logs_p, logw, logw_, x_mask, y_mask = self.model_g(
            x, x_lengths, mel, mel_lengths, sid=sid,
        )

        # exact NLL of z under N(m_p, exp(logs_p)^2), minus the flow's log-det
        # Jacobian (change-of-variables formula, GlowTTS paper eq. 2-3),
        # normalized per element and including the 0.5*log(2*pi) constant so
        # loss values are comparable with the reference implementations.
        num_elements = torch.sum(y_mask) * self.hparams.mel_channels
        l_mle = torch.sum(logs_p) + 0.5 * torch.sum(torch.exp(-2 * logs_p) * (z - m_p) ** 2)
        l_mle = l_mle / num_elements - torch.sum(logdet) / num_elements
        l_mle = l_mle + 0.5 * math.log(2 * math.pi)

        l_dur = torch.sum((logw - logw_) ** 2) / torch.sum(x_lengths)

        loss = l_mle + self.hparams.c_dur * l_dur
        return loss, l_mle, l_dur

    def training_step(self, batch: Batch, batch_idx: int):
        loss, l_mle, l_dur = self._compute_loss(batch)
        self.log("loss_mle", l_mle)
        self.log("loss_dur", l_dur)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        loss, l_mle, l_dur = self._compute_loss(batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        # Noam warmup as in the reference recipe (glow-tts commons.Adam:
        # lr = scale * dim^-0.5 * min(step^-0.5, step * warmup^-1.5),
        # betas (0.9, 0.98), eps 1e-9, warmup 4000). Plain Adam — the
        # reference applies no weight decay. The schedule here is
        # normalized so `learning_rate` is the post-warmup *peak*; the
        # reference peak with dim=192, warmup=4000 is
        # 192^-0.5 * 4000^-0.5 ~= 1.14e-3, matching the 1e-3 default.
        opt = torch.optim.Adam(
            self.model_g.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
            eps=self.hparams.eps,
        )
        warmup = max(1, self.hparams.warmup_steps)

        def noam(step: int) -> float:
            step = max(1, step)
            # scale relative to the configured LR so learning_rate remains the
            # effective post-warmup ceiling
            return min(step ** -0.5, step * warmup ** -1.5) * warmup ** 0.5

        sched = torch.optim.lr_scheduler.LambdaLR(opt, noam)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}
