"""pytorch_lightning training loops for the three SuperTonic stages.

Each stage is a separate :class:`~pytorch_lightning.LightningModule`:

* :class:`AutoencoderModule` — GAN training of the speech autoencoder (manual
  optimization: a generator step and a discriminator step per batch). At the end
  of every epoch it measures the per-channel latent statistics over up to
  ``stats_batches`` batches and stores them in the autoencoder's buffers, so the
  checkpoint carries the normalisation the later stages depend on.
* :class:`TextToLatentModule` — conditional flow matching, with target latents
  produced on the fly by a frozen autoencoder.
* :class:`DurationPredictorModule` — regress the utterance duration (in seconds)
  from text plus a reference crop of the utterance's own latent.

All three take a :class:`SuperTonicConfig` and a :class:`CharTokenizer` (or the
data needed to build one) so a checkpoint fully reconstructs the model. Datasets
are the filelist form in :mod:`phoonnx_train.supertonic.dataset`. Lightning
already round-trips model/optimizer/scheduler/step in its checkpoints; the
config and tokenizer are embedded via :meth:`on_save_checkpoint` so ONNX export
and fine-tuning can rebuild the exact model.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch

from phoonnx_train.supertonic.config import SuperTonicConfig
from phoonnx_train.supertonic.text import CharTokenizer

LOG = logging.getLogger(__name__)


def _build_config(config: Any) -> SuperTonicConfig:
    if isinstance(config, SuperTonicConfig):
        return config
    if isinstance(config, dict):
        cfg = SuperTonicConfig()
        for stage in ("ae", "ttl", "dp"):
            for k, v in (config.get(stage) or {}).items():
                if hasattr(getattr(cfg, stage), k):
                    setattr(getattr(cfg, stage), k, tuple(v) if isinstance(v, list) else v)
        cfg.vocab_size = config.get("vocab_size", cfg.vocab_size)
        return cfg
    return SuperTonicConfig()


class _StageBase(pl.LightningModule):
    stage_name = "base"

    def _stash(self, config: SuperTonicConfig, tokenizer: Optional[CharTokenizer]):
        self._config = config
        self._tokenizer = tokenizer or CharTokenizer()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["supertonic_config"] = asdict(self._config)
        checkpoint["supertonic_tokenizer"] = self._tokenizer.to_dict()
        checkpoint["supertonic_stage"] = self.stage_name

    def _load_into(self, module, state_dict, prefix, source):
        picked = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if not picked:
            picked = state_dict
        missing = set(module.state_dict()) - set(picked)
        if missing:
            raise ValueError(
                f"{source} does not contain weights for {type(module).__name__}: "
                f"{len(missing)} missing key(s), e.g. {sorted(missing)[:3]}")
        module.load_state_dict(picked, strict=False)

    def _make_loader(self, ds, batch_size, shuffle, collate, num_workers):
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle,
            collate_fn=collate, num_workers=num_workers,
            pin_memory=torch.cuda.is_available())


# ----------------------------------------------------------------------
# Stage 1 — speech autoencoder (GAN)
# ----------------------------------------------------------------------

class AutoencoderModule(_StageBase):
    stage_name = "autoencoder"

    def __init__(self, config: Any = None, dataset: Optional[List[str]] = None,
                 root_dir: str = ".", batch_size: int = 4, num_workers: int = 0,
                 segment_seconds: float = 0.5, base_lr: float = 2e-4,
                 lambda_mel: float = 45.0, lambda_fm: float = 2.0,
                 stats_batches: int = 200, **_ignored: Any):
        super().__init__()
        from phoonnx_train.supertonic.autoencoder import SpeechAutoencoder
        from phoonnx_train.supertonic.discriminators import (
            MultiPeriodDiscriminator,
            MultiResolutionDiscriminator,
        )
        from phoonnx_train.supertonic.losses import MultiResolutionMelLoss

        cfg = _build_config(config)
        self._stash(cfg, None)
        self.save_hyperparameters(ignore=["config"])
        self.automatic_optimization = False

        self.generator = SpeechAutoencoder(cfg.ae)
        self.mpd = MultiPeriodDiscriminator()
        self.mrd = MultiResolutionDiscriminator()
        self.mel_loss = MultiResolutionMelLoss(cfg.ae.sample_rate)
        self._dataset = dataset
        self._root = root_dir
        self._train_set = None

    def setup(self, stage: Optional[str] = None):
        from phoonnx_train.supertonic.dataset import WaveformDataset
        if self._train_set is not None or not self._dataset:
            return
        seg = int(self.hparams.segment_seconds * self._config.ae.sample_rate)
        self._train_set = torch.utils.data.ConcatDataset(
            [WaveformDataset(p, self._root, self._config.ae.sample_rate, seg) for p in self._dataset])

    def train_dataloader(self):
        from phoonnx_train.supertonic.dataset import waveform_collate
        return self._make_loader(self._train_set, self.hparams.batch_size, True,
                                 waveform_collate, self.hparams.num_workers)

    def _disc_outputs(self, wav):
        mpd_out, mpd_feat = self.mpd(wav)
        mrd_out, mrd_feat = self.mrd(wav)
        return mpd_out + mrd_out, mpd_feat + mrd_feat

    def training_step(self, batch, batch_idx):
        from phoonnx_train.supertonic.losses import (
            discriminator_loss,
            feature_matching_loss,
            generator_adv_loss,
        )
        opt_g, opt_d = self.optimizers()
        wav = batch
        recon, _ = self.generator(wav)
        n = min(wav.shape[-1], recon.shape[-1])
        wav, recon = wav[..., :n], recon[..., :n]

        # discriminator step
        real_out, _ = self._disc_outputs(wav)
        fake_out, _ = self._disc_outputs(recon.detach())
        d_loss = discriminator_loss(real_out, fake_out)
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        # generator step
        real_out, real_feat = self._disc_outputs(wav)
        fake_out, fake_feat = self._disc_outputs(recon)
        g_adv = generator_adv_loss(fake_out)
        g_fm = feature_matching_loss(real_feat, fake_feat)
        g_mel = self.mel_loss(recon, wav)
        g_loss = g_adv + self.hparams.lambda_fm * g_fm + self.hparams.lambda_mel * g_mel
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        self.log_dict({"train/g_loss": g_loss, "train/d_loss": d_loss, "train/mel": g_mel},
                      prog_bar=True, batch_size=wav.shape[0])
        return g_loss

    def on_train_epoch_end(self):
        """Refit the per-channel latent statistics the downstream stages normalise with.

        They must describe the encoder as it stands in the checkpoint about to be
        written, so they are measured in a separate encode-only pass rather than
        accumulated across an epoch of changing weights.
        """
        loader = self.trainer.train_dataloader if self.trainer else None
        if loader is None:
            return
        was_training = self.generator.training
        self.generator.eval()
        latents = []
        with torch.no_grad():
            for i, wav in enumerate(loader):
                if i >= self.hparams.stats_batches:
                    break
                latents.append(self.generator.encode(wav.to(self.device)).cpu())
        self.generator.train(was_training)
        if latents:
            self.generator.fit_latent_stats(latents)

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=self.hparams.base_lr, betas=(0.8, 0.99))
        opt_d = torch.optim.AdamW(
            list(self.mpd.parameters()) + list(self.mrd.parameters()),
            lr=self.hparams.base_lr, betas=(0.8, 0.99))
        return [opt_g, opt_d]


# ----------------------------------------------------------------------
# frozen-AE helper for stages 2 and 3
# ----------------------------------------------------------------------

class _NeedsFrozenAE(_StageBase):
    def _init_ae(self, cfg, ae_checkpoint):
        from phoonnx_train.supertonic.autoencoder import SpeechAutoencoder
        ae = SpeechAutoencoder(cfg.ae)
        if not ae_checkpoint:
            raise ValueError(f"stage {self.stage_name!r} needs a trained autoencoder checkpoint "
                             "(ae_checkpoint) to produce latent targets")
        ckpt = torch.load(ae_checkpoint, map_location="cpu", weights_only=False)
        found = ckpt.get("supertonic_stage")
        if found is not None and found != AutoencoderModule.stage_name:
            raise ValueError(f"{ae_checkpoint} is a {found!r} checkpoint, not an autoencoder one")
        self._load_into(ae, ckpt.get("state_dict", ckpt), "generator.", ae_checkpoint)
        if not bool(ae.stats_fitted):
            raise ValueError(
                f"{ae_checkpoint} carries no fitted latent statistics; stage {self.stage_name!r} "
                "would train against unnormalised latents. Train the autoencoder to the end of an "
                "epoch so its statistics are fitted and saved.")
        ae.eval()
        for p in ae.parameters():
            p.requires_grad_(False)
        self.frozen_ae = ae

    @torch.no_grad()
    def _targets(self, wav, wav_lens, stage_cfg):
        from phoonnx_train.supertonic.latent_utils import normalize_and_compress
        from phoonnx_train.supertonic.layers import make_mask
        raw = self.frozen_ae.encode(wav)  # (B, ld, T)
        z1 = normalize_and_compress(self.frozen_ae, raw, stage_cfg.compress_factor, stage_cfg.normalizer_scale)
        # per-sample compressed-frame lengths
        # centred STFT emits one frame per hop plus one
        frames = torch.div(wav_lens, self._config.ae.hop_length, rounding_mode="floor") + 1
        latent_lens = torch.div(frames, stage_cfg.compress_factor, rounding_mode="floor").clamp(min=1, max=z1.shape[-1])
        return z1, make_mask(latent_lens, z1.shape[-1]), latent_lens


# ----------------------------------------------------------------------
# Stage 2 — text to latent (flow matching)
# ----------------------------------------------------------------------

class TextToLatentModule(_NeedsFrozenAE):
    stage_name = "text_to_latent"

    def __init__(self, config: Any = None, tokenizer: Any = None,
                 dataset: Optional[List[str]] = None, root_dir: str = ".",
                 ae_checkpoint: Optional[str] = None, batch_size: int = 4,
                 num_workers: int = 0, base_lr: float = 1e-4, **_ignored: Any):
        super().__init__()
        from phoonnx_train.supertonic.text_to_latent import TextToLatentModel

        cfg = _build_config(config)
        tok = tokenizer if isinstance(tokenizer, CharTokenizer) else (
            CharTokenizer.from_dict(tokenizer) if isinstance(tokenizer, dict) else CharTokenizer())
        self._stash(cfg, tok)
        self.save_hyperparameters(ignore=["config", "tokenizer"])

        self.model = TextToLatentModel(cfg.ttl, cfg.vocab_size)
        self._init_ae(cfg, ae_checkpoint)
        self._dataset, self._root, self._ae_ckpt = dataset, root_dir, ae_checkpoint
        self._train_set = None

    def setup(self, stage: Optional[str] = None):
        from phoonnx_train.supertonic.dataset import TextAudioDataset
        if self._train_set is not None or not self._dataset:
            return
        self._train_set = torch.utils.data.ConcatDataset([
            TextAudioDataset(p, self._root, self._tokenizer, self._config.ae.sample_rate)
            for p in self._dataset])

    def train_dataloader(self):
        from phoonnx_train.supertonic.dataset import text_audio_collate
        return self._make_loader(self._train_set, self.hparams.batch_size, True,
                                 text_audio_collate, self.hparams.num_workers)

    def _loss(self, batch):
        from phoonnx_train.supertonic.latent_utils import sample_reference_crop
        from phoonnx_train.supertonic.layers import make_mask
        from phoonnx_train.supertonic.text_to_latent import flow_matching_loss
        wav, wav_lens, text_ids, text_lens = batch
        cfg = self._config.ttl
        z1, latent_mask, latent_lens = self._targets(wav, wav_lens, cfg)
        frame_rate = self._config.ae.sample_rate / self._config.ae.hop_length / cfg.compress_factor
        ref, ref_mask, ref_time_mask = sample_reference_crop(z1, latent_lens, frame_rate)
        text_mask = make_mask(text_lens.clamp(min=1), text_ids.shape[-1])
        return flow_matching_loss(self.model, z1, latent_mask, text_ids, text_mask,
                                  ref, ref_mask, ref_time_mask, n_expand=cfg.batch_expand)

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log("train/fm_loss", loss, prog_bar=True, batch_size=batch[0].shape[0])
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.base_lr, betas=(0.9, 0.98))


# ----------------------------------------------------------------------
# Stage 3 — duration predictor
# ----------------------------------------------------------------------

class DurationPredictorModule(_NeedsFrozenAE):
    stage_name = "duration_predictor"

    def __init__(self, config: Any = None, tokenizer: Any = None,
                 dataset: Optional[List[str]] = None, root_dir: str = ".",
                 ae_checkpoint: Optional[str] = None, batch_size: int = 4,
                 num_workers: int = 0, base_lr: float = 1e-4, **_ignored: Any):
        super().__init__()
        from phoonnx_train.supertonic.duration_predictor import DurationPredictor

        cfg = _build_config(config)
        tok = tokenizer if isinstance(tokenizer, CharTokenizer) else (
            CharTokenizer.from_dict(tokenizer) if isinstance(tokenizer, dict) else CharTokenizer())
        self._stash(cfg, tok)
        self.save_hyperparameters(ignore=["config", "tokenizer"])

        self.model = DurationPredictor(cfg.dp, cfg.vocab_size)
        self._init_ae(cfg, ae_checkpoint)
        self._dataset, self._root = dataset, root_dir
        self._train_set = None

    def setup(self, stage: Optional[str] = None):
        from phoonnx_train.supertonic.dataset import TextAudioDataset
        if self._train_set is not None or not self._dataset:
            return
        self._train_set = torch.utils.data.ConcatDataset([
            TextAudioDataset(p, self._root, self._tokenizer, self._config.ae.sample_rate)
            for p in self._dataset])

    def train_dataloader(self):
        from phoonnx_train.supertonic.dataset import text_audio_collate
        return self._make_loader(self._train_set, self.hparams.batch_size, True,
                                 text_audio_collate, self.hparams.num_workers)

    def _loss(self, batch):
        from phoonnx_train.supertonic.duration_predictor import duration_loss
        from phoonnx_train.supertonic.latent_utils import sample_reference_crop
        from phoonnx_train.supertonic.layers import make_mask
        wav, wav_lens, text_ids, text_lens = batch
        cfg = self._config.dp
        z1, _, latent_lens = self._targets(wav, wav_lens, cfg)
        frame_rate = self._config.ae.sample_rate / self._config.ae.hop_length / cfg.compress_factor
        ref, ref_mask, _ = sample_reference_crop(z1, latent_lens, frame_rate)
        text_mask = make_mask(text_lens.clamp(min=1), text_ids.shape[-1])
        pred = self.model(text_ids, text_mask, ref, ref_mask)
        target = wav_lens.float() / self._config.ae.sample_rate
        return duration_loss(pred, target)

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log("train/dur_loss", loss, prog_bar=True, batch_size=batch[0].shape[0])
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.base_lr, betas=(0.9, 0.98))
