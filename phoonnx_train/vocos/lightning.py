"""Lightning module for Vocos vocoder GAN training.

Follows the training recipe of the Vocos paper (Siuzdak, 2023): AdamW for
generator and discriminators, hinge adversarial loss, feature matching and
an L1 log-mel reconstruction loss.  Uses manual optimization (works on
both Lightning 1.9 and 2.x) so the two optimizers step in the standard
D-then-G order every batch.
"""
import logging
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch

from phoonnx_train.vits.mel_processing import mel_spectrogram_torch
from phoonnx_train.vocos.data import MelConfig, parse_warm_start
from phoonnx_train.vocos.dataset import AudioCropDataset
from phoonnx_train.vocos.models import (
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    VocosGenerator,
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
)

_LOGGER = logging.getLogger(__name__)


class VocosTrainingModule(pl.LightningModule):
    """GAN training for a Vocos mel→waveform generator."""

    def __init__(
        self,
        mel: Optional[dict] = None,
        dim: int = 512,
        intermediate_dim: int = 1536,
        num_layers: int = 8,
        learning_rate: float = 5e-4,
        betas: Tuple[float, float] = (0.8, 0.9),
        mel_loss_coeff: float = 45.0,
        batch_size: int = 16,
        num_workers: int = 4,
        crop_samples: int = 22016,
        audio_files: Optional[List] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["audio_files"])
        # hyperparameters must stay yaml-serializable, so mel travels as a
        # plain dict and is materialized here
        self.mel = MelConfig(**mel) if mel else MelConfig()
        self.audio_files = audio_files or []
        self.crop_samples = crop_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.betas = betas
        self.mel_loss_coeff = mel_loss_coeff

        self.generator = VocosGenerator(
            n_mels=self.mel.n_mels,
            n_fft=self.mel.n_fft,
            hop_length=self.mel.hop_length,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
        )
        self.mpd = MultiPeriodDiscriminator()
        self.mrd = MultiResolutionDiscriminator()

        self.automatic_optimization = False

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    def _mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Log-mel with the exact acoustic-model settings (no drift allowed)."""
        m = self.mel
        return mel_spectrogram_torch(
            audio, m.n_fft, m.n_mels, m.sample_rate, m.hop_length,
            m.win_length, m.fmin, m.fmax,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        audio = batch
        opt_g, opt_d = self.optimizers()

        mel = self._mel(audio)
        audio_hat = self.generator(mel)
        # center-padded ISTFT returns (T-1)*hop samples; align both signals
        n = min(audio.shape[-1], audio_hat.shape[-1])
        audio, audio_hat = audio[..., :n], audio_hat[..., :n]

        # --- discriminators ---
        real_p, _ = self.mpd(audio)
        fake_p, _ = self.mpd(audio_hat.detach())
        real_r, _ = self.mrd(audio)
        fake_r, _ = self.mrd(audio_hat.detach())
        loss_d = discriminator_loss(real_p, fake_p) + discriminator_loss(real_r, fake_r)

        opt_d.zero_grad()
        self.manual_backward(loss_d)
        opt_d.step()

        # --- generator ---
        real_p, fmap_real_p = self.mpd(audio)
        fake_p, fmap_fake_p = self.mpd(audio_hat)
        real_r, fmap_real_r = self.mrd(audio)
        fake_r, fmap_fake_r = self.mrd(audio_hat)

        loss_mel = torch.nn.functional.l1_loss(self._mel(audio_hat), self._mel(audio))
        loss_adv = generator_adversarial_loss(fake_p) + generator_adversarial_loss(fake_r)
        loss_fm = feature_matching_loss(fmap_real_p, fmap_fake_p) + feature_matching_loss(
            fmap_real_r, fmap_fake_r
        )
        loss_g = loss_adv + loss_fm + self.mel_loss_coeff * loss_mel

        opt_g.zero_grad()
        self.manual_backward(loss_g)
        opt_g.step()

        self.log_dict(
            {
                "loss_disc": loss_d,
                "loss_gen": loss_g,
                "loss_mel": loss_mel,
                "loss_fm": loss_fm,
                "loss_adv": loss_adv,
            },
            prog_bar=True,
        )
        return None

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(
            self.generator.parameters(), lr=self.learning_rate, betas=self.betas
        )
        opt_d = torch.optim.AdamW(
            list(self.mpd.parameters()) + list(self.mrd.parameters()),
            lr=self.learning_rate,
            betas=self.betas,
        )
        return [opt_g, opt_d]

    def train_dataloader(self):
        dataset = AudioCropDataset(
            files=self.audio_files,
            mel=self.mel,
            crop_samples=self.crop_samples,
            train=True,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    # ------------------------------------------------------------------
    # Warm start
    # ------------------------------------------------------------------

    def warm_start(self, source: str) -> int:
        """Initialize the generator from a pretrained Vocos checkpoint.

        ``source`` is a local ``.ckpt``/``.bin`` file or a HuggingFace repo
        id (e.g. a published ``vocos-mel-22khz`` style repo).  Parameters
        are copied by name where the shapes match; everything else keeps
        its fresh initialization.  Returns the number of matched tensors.
        """
        kind, value = parse_warm_start(source)
        if kind == "hf":
            from huggingface_hub import hf_hub_download

            value = hf_hub_download(repo_id=value, filename="pytorch_model.bin")
        state = torch.load(value, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # Map reference-Vocos layouts onto the vendored module names.
        # Inference weights use bare "backbone." / "head." prefixes; full
        # training checkpoints prefix the discriminators as
        # "multiperioddisc." / "multiresddisc.".
        remap = {
            "backbone.": "generator.backbone.",
            "head.": "generator.head.",
            "multiperioddisc.": "mpd.",
            "multiresddisc.": "mrd.",
        }
        renamed = {}
        for key, tensor in state.items():
            for old, new in remap.items():
                if key.startswith(old):
                    key = new + key[len(old):]
                    break
            renamed[key] = tensor

        matched, total = 0, 0
        for prefix, module in (("generator.", self.generator),
                               ("mpd.", self.mpd),
                               ("mrd.", self.mrd)):
            own = module.state_dict()
            total += len(own)
            incoming = {
                key.removeprefix(prefix): tensor
                for key, tensor in renamed.items()
                if key.startswith(prefix)
                and key.removeprefix(prefix) in own
                and own[key.removeprefix(prefix)].shape == tensor.shape
            }
            matched += len(incoming)
            own.update(incoming)
            module.load_state_dict(own)

        _LOGGER.info(
            "Warm start from %s: matched %d/%d tensors (%.1f%%)",
            source, matched, total, 100.0 * matched / max(total, 1),
        )
        return matched
