"""LightningModule + dataset/collate wrapping the vendored Mixer-TTS model.

The model (``phoonnx_train/mixertts/models``) is a self-contained port of
NVIDIA NeMo's Mixer-TTS (Apache-2.0,
nemo/collections/tts/models/mixer_tts.py @ 7256db1; paper: Tatanov et al.,
2021, https://arxiv.org/abs/2110.03584) with the speaker/emotion/energy
conditioning and optional LSGAN mel-patch refinement added by
nipponjo/tts-arabic-pytorch (MIT).

Training reuses the shared phoonnx preprocessing pipeline (same as the
FastPitch engine): phoneme ids + linear-spectrogram caches come from
``phoonnx_train.preprocess`` (``dataset.jsonl``), the mel target is derived
from the linear spectrogram at load time via
``phoonnx_train.vits.mel_processing`` (fmin/fmax pinned to the shared
0/8000 Hz convention), and the frame-aligned F0 target is read from the
``<utterance>.f0-<method>.npy`` sidecars written by the engine's
``extra_preprocess`` (``librosa.pyin`` at the mel hop, on the same
trimmed/normalized cached audio the mels come from).

Loss/optimizer fidelity vs upstream NeMo:

- losses: masked-MSE mel/log-duration/pitch/energy + ForwardSum (CTC)
  aligner loss + delayed, ramped binarization (bin) loss — identical
  formulation and scales (durs 0.1 / mel 1.0 / pitch 0.1 / energy 0.1,
  bin ramp from ``bin_loss_start_ratio * max_epochs``), computed by the
  vendored model's ``_metrics``.
- alignment: unsupervised AlignmentEncoder + monotonic alignment search
  (``binarize_attention_parallel``) with the beta-binomial prior, as in
  the one-TTS-alignment recipe (https://arxiv.org/abs/2108.10447).
- optimizer: AdamW(lr, betas=(0.9, 0.98), weight_decay=1e-6) — the betas
  follow nipponjo's non-GAN recipe — under a Noam warmup schedule matching
  NeMo's NoamAnnealing (warmup 1000 steps, d_model=1: peak effective LR is
  ``lr / sqrt(warmup)``).
- optional (off by default, non-NeMo) LSGAN PatchDiscriminator on random
  mel chunks + feature-matching loss, following nipponjo; this switches
  the module to manual optimization with a second AdamW.
"""
import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch import LongTensor
from torch.utils.data import DataLoader, Dataset, random_split

from phoonnx_train.fastpitch.pitch_stats import (
    f0_cache_path,
    load_or_compute_pitch_stats,
)
from phoonnx_train.mixertts.models.common.loss import (
    PatchDiscriminator,
    calc_feature_match_loss,
    extract_chunks,
)
from phoonnx_train.mixertts.models.mixer_tts.mixer_tts import (
    MixerTTSModel as _MixerTTS,
)
from phoonnx_train.mixertts.prior import BetaBinomialPrior
from phoonnx_train.vits.dataset import PhoonnxDataset, Utterance
from phoonnx_train.vits.mel_processing import spec_to_mel_torch

_LOG = logging.getLogger(__name__)


class UtteranceTensors:
    __slots__ = ("phoneme_ids", "mel", "speaker_id", "pitch", "energy", "attn_prior")

    def __init__(self, phoneme_ids, mel, speaker_id, pitch, energy, attn_prior):
        self.phoneme_ids = phoneme_ids
        self.mel = mel
        self.speaker_id = speaker_id
        self.pitch = pitch
        self.energy = energy
        self.attn_prior = attn_prior


class Batch:
    __slots__ = (
        "phoneme_ids", "phoneme_lengths", "mels", "mel_lengths",
        "pitch", "energy", "speaker_ids", "attn_prior",
    )

    def __init__(self, phoneme_ids, phoneme_lengths, mels, mel_lengths,
                 pitch, energy, speaker_ids, attn_prior):
        self.phoneme_ids = phoneme_ids
        self.phoneme_lengths = phoneme_lengths
        self.mels = mels
        self.mel_lengths = mel_lengths
        self.pitch = pitch
        self.energy = energy
        self.speaker_ids = speaker_ids
        self.attn_prior = attn_prior


class MixerTTSDataset(Dataset):
    """Wraps the shared ``PhoonnxDataset`` utterances for Mixer-TTS.

    Per utterance: phoneme ids, log-mel target (from the shared linear
    spectrogram cache), frame-aligned z-scored F0, per-frame energy
    (L2 norm over mel bins, as in FastPitch/nipponjo), and the
    beta-binomial alignment prior.
    """

    def __init__(self, dataset_paths: List[Path], mel_channels: int,
                 filter_length: int, sample_rate: int,
                 mel_fmin: float, mel_fmax: Optional[float],
                 max_phoneme_ids: Optional[int] = None,
                 f0_method: str = "pyin"):
        jsonl_paths = [
            (Path(p) / "dataset.jsonl") if Path(p).is_dir() else Path(p)
            for p in dataset_paths
        ]
        self._inner = PhoonnxDataset(jsonl_paths, max_phoneme_ids=max_phoneme_ids)
        self.mel_channels = mel_channels
        self.filter_length = filter_length
        self.sample_rate = sample_rate
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.f0_method = f0_method
        self.prior = BetaBinomialPrior()
        self.pitch_mean, self.pitch_std = load_or_compute_pitch_stats(
            dataset_paths,
            [f0_cache_path(utt.audio_spec_path, method=f0_method) for utt in self._inner.utterances],
            method=f0_method,
        )

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> UtteranceTensors:
        utt: Utterance = self._inner.utterances[idx]
        spec = torch.load(utt.audio_spec_path)  # [n_fft//2+1, T]
        mel = spec_to_mel_torch(
            spec.unsqueeze(0), self.filter_length, self.mel_channels,
            self.sample_rate, self.mel_fmin, self.mel_fmax,
        ).squeeze(0)  # [mel_channels, T_mel]
        t_mel = mel.size(1)

        pitch = torch.zeros(t_mel)
        f0_candidate = f0_cache_path(utt.audio_spec_path, method=self.f0_method)
        if f0_candidate.exists():
            f0 = np.load(f0_candidate).astype("float32")
            # length-reconcile to T_mel (extra_preprocess already keeps the
            # drift within ±2 frames); z-score voiced frames with the corpus
            # stats, unvoiced frames stay 0 — NeMo's pitch normalization
            f0 = f0[:t_mel]
            voiced = f0 > 0
            f0[voiced] = (f0[voiced] - self.pitch_mean) / self.pitch_std
            pitch[: len(f0)] = torch.from_numpy(f0)

        # per-frame energy: L2 norm over mel bins (FastPitch recipe); the
        # model log-compresses the duration-averaged target itself
        energy = torch.norm(mel.float(), dim=0, p=2)  # [T_mel]
        attn_prior = torch.from_numpy(
            self.prior(t_mel, len(utt.phoneme_ids))
        )  # [T_mel, T_text]

        return UtteranceTensors(
            phoneme_ids=LongTensor(utt.phoneme_ids),
            mel=mel,
            speaker_id=LongTensor([utt.speaker_id]) if utt.speaker_id is not None else None,
            pitch=pitch,
            energy=energy,
            attn_prior=attn_prior,
        )


class MixerTTSCollate:
    def __init__(self, is_multispeaker: bool):
        self.is_multispeaker = is_multispeaker

    def __call__(self, utterances: List[UtteranceTensors]) -> Batch:
        n = len(utterances)
        max_ph = max(u.phoneme_ids.size(0) for u in utterances)
        max_mel = max(u.mel.size(1) for u in utterances)
        mel_channels = utterances[0].mel.size(0)

        phoneme_ids = torch.zeros(n, max_ph, dtype=torch.long)
        phoneme_lengths = torch.zeros(n, dtype=torch.long)
        mels = torch.zeros(n, mel_channels, max_mel)
        mel_lengths = torch.zeros(n, dtype=torch.long)
        pitch = torch.zeros(n, max_mel)
        energy = torch.zeros(n, max_mel)
        attn_prior = torch.zeros(n, max_mel, max_ph)
        speaker_ids = torch.zeros(n, dtype=torch.long) if self.is_multispeaker else None

        for i, utt in enumerate(utterances):
            pl_ = utt.phoneme_ids.size(0)
            ml = utt.mel.size(1)
            phoneme_ids[i, :pl_] = utt.phoneme_ids
            phoneme_lengths[i] = pl_
            mels[i, :, :ml] = utt.mel
            mel_lengths[i] = ml
            pitch[i, :ml] = utt.pitch[:ml]
            energy[i, :ml] = utt.energy[:ml]
            attn_prior[i, :ml, :pl_] = utt.attn_prior
            if speaker_ids is not None and utt.speaker_id is not None:
                speaker_ids[i] = utt.speaker_id

        return Batch(phoneme_ids, phoneme_lengths, mels, mel_lengths,
                     pitch, energy, speaker_ids, attn_prior)


class MixerTTSModule(pl.LightningModule):
    """LightningModule wrapping the vendored Mixer-TTS model."""

    def __init__(
        self,
        num_symbols: int,
        num_speakers: int = 1,
        sample_rate: int = 22050,
        filter_length: int = 1024,
        mel_channels: int = 80,
        mel_fmin: float = 0.0,
        # pinned to the shared phoonnx mel convention: defaulting to Nyquist
        # would silently mismatch every shared vocoder
        mel_fmax: Optional[float] = 8000.0,
        symbols_embedding_dim: int = 384,
        num_emotions: int = 1,
        dataset: Optional[List[Path]] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        warmup_steps: int = 1000,
        train_gan: bool = False,
        gan_chunk_len: int = 128,
        gan_learning_rate: float = 1e-4,
        batch_size: int = 8,
        num_workers: int = 1,
        validation_split: float = 0.1,
        num_test_examples: int = 5,
        max_phoneme_ids: Optional[int] = None,
        # bin (hard-attention) loss is delayed then ramped — hard attention
        # against a still-random soft alignment destabilizes the aligner
        bin_loss_start_ratio: float = 0.2,
        bin_loss_warmup_epochs: int = 100,
        # corpus F0 stats; filled from the dataset's pitch_stats.json at
        # dataset load and persisted in the checkpoint hparams so export /
        # inference know the pitch-normalization domain
        pitch_mean: float = 0.0,
        pitch_std: float = 1.0,
        # F0 extraction method used at preprocessing time — "pyin" (default)
        # or "dio"/"harvest" (WORLD via pyworld, train-pyworld extra). Must
        # match whatever ``--f0-method`` extra_preprocess ran with, since it
        # selects which ``<utterance>.f0-<method>.npy`` sidecar to read.
        f0_method: str = "pyin",
        **kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = _MixerTTS(
            n_mel_channels=mel_channels,
            num_tokens=num_symbols,
            padding_idx=0,
            symbols_embedding_dim=symbols_embedding_dim,
            n_speakers=num_speakers,
            n_emotions=num_emotions,
        )
        self.model.pitch_mean = pitch_mean
        self.model.pitch_std = pitch_std
        self.model.add_bin_loss = False
        self.model.bin_loss_scale = 0.0
        self.model.bin_loss_start_ratio = bin_loss_start_ratio
        self.model.bin_loss_warmup_epochs = bin_loss_warmup_epochs

        self.train_gan = train_gan
        self.critic = PatchDiscriminator(1, 32) if train_gan else None
        self.tar_len = gan_chunk_len
        if train_gan:
            # two optimizers (generator + critic) — upstream GAN loop style
            self.automatic_optimization = False

        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
        self._load_datasets(validation_split, num_test_examples, max_phoneme_ids)

    # ------------------------------------------------------------------
    def _load_datasets(self, validation_split: float, num_test_examples: int,
                       max_phoneme_ids: Optional[int]) -> None:
        if not self.hparams.dataset:
            _LOG.debug("No dataset to load")
            return
        full_dataset = MixerTTSDataset(
            self.hparams.dataset,
            mel_channels=self.hparams.mel_channels,
            filter_length=self.hparams.filter_length,
            sample_rate=self.hparams.sample_rate,
            mel_fmin=self.hparams.mel_fmin,
            mel_fmax=self.hparams.mel_fmax,
            max_phoneme_ids=max_phoneme_ids,
            f0_method=self.hparams.f0_method,
        )
        # persist the corpus pitch stats (checkpointed via hparams) so
        # inference-time pitch controls know the normalization domain
        self.hparams.pitch_mean = full_dataset.pitch_mean
        self.hparams.pitch_std = full_dataset.pitch_std
        self.model.pitch_mean = full_dataset.pitch_mean
        self.model.pitch_std = full_dataset.pitch_std
        valid_size = max(0, int(len(full_dataset) * validation_split))
        test_size = min(num_test_examples, max(0, len(full_dataset) - valid_size))
        train_size = len(full_dataset) - valid_size - test_size
        self._train_dataset, self._test_dataset, self._val_dataset = random_split(
            full_dataset, [train_size, test_size, valid_size]
        )

    def _collate(self) -> MixerTTSCollate:
        return MixerTTSCollate(is_multispeaker=self.hparams.num_speakers > 1)

    def train_dataloader(self):
        return DataLoader(self._train_dataset, collate_fn=self._collate(),
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self._val_dataset, collate_fn=self._collate(),
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self._test_dataset, collate_fn=self._collate(),
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.batch_size)

    # ------------------------------------------------------------------
    def forward(self, x, pace: float = 1.0, speaker: int = 0, emotion: int = 0):
        return self.model.infer(x, pace=pace, speaker=speaker, emotion=emotion)

    def on_train_epoch_start(self):
        if self.train_gan:
            # GAN refinement phase: the aligner is assumed converged — bin
            # loss at full scale from the start, no ramp restart (nipponjo)
            self.model.add_bin_loss = True
            self.model.bin_loss_scale = 1.0
            return
        # the vendored model defines this schedule but is a plain nn.Module,
        # so Lightning never calls its hook — drive it from here (mirrors
        # NeMo MixerTTSModel.on_train_epoch_start)
        max_epochs = max(1, self.trainer.max_epochs or 1)
        start_epoch = int(np.ceil(self.model.bin_loss_start_ratio * max_epochs))
        if not self.model.add_bin_loss and self.current_epoch >= start_epoch:
            _LOG.info("Using hard attentions after epoch: %d", self.current_epoch)
            self.model.add_bin_loss = True
        if self.model.add_bin_loss:
            self.model.bin_loss_scale = min(
                (self.current_epoch - start_epoch)
                / max(1, self.model.bin_loss_warmup_epochs), 1.0)

    def _step(self, batch: Batch):
        # no emotion labels in the shared dataset format — condition on the
        # neutral (0) emotion when an emotion embedding is configured
        emotion = None
        if self.model.emotion_emb is not None:
            emotion = torch.zeros(batch.phoneme_ids.size(0), dtype=torch.long,
                                  device=batch.phoneme_ids.device)
        preds = self.model(
            text=batch.phoneme_ids,
            text_len=batch.phoneme_lengths,
            pitch=batch.pitch,
            energy=batch.energy,
            spect=batch.mels,
            spect_len=batch.mel_lengths,
            attn_prior=batch.attn_prior,
            speaker=batch.speaker_ids,
            emotion=emotion,
        )
        (pred_spect, _, pred_log_durs, pred_pitch, pred_energy,
         attn_soft, attn_logprob, attn_hard, attn_hard_dur) = preds
        (loss, durs_loss, acc, acc_dist_1, acc_dist_3, pitch_loss,
         energy_loss, mel_loss, ctc_loss, bin_loss) = self.model._metrics(
            pred_durs=pred_log_durs,
            pred_pitch=pred_pitch,
            pred_energy=pred_energy,
            true_durs=attn_hard_dur,
            true_text_len=batch.phoneme_lengths,
            true_pitch=batch.pitch,
            true_energy=batch.energy,
            true_spect=batch.mels,
            pred_spect=pred_spect,
            true_spect_len=batch.mel_lengths,
            attn_logprob=attn_logprob,
            attn_soft=attn_soft,
            attn_hard=attn_hard,
            attn_hard_dur=attn_hard_dur,
        )
        losses = {
            "loss": loss, "durs_loss": durs_loss, "mel_loss": mel_loss,
            "pitch_loss": pitch_loss, "energy_loss": energy_loss,
            "ctc_loss": ctc_loss,
        }
        if bin_loss is not None:
            losses["bin_loss"] = bin_loss
        return losses, pred_spect

    def training_step(self, batch: Batch, batch_idx: int):
        if not self.train_gan:
            losses, _ = self._step(batch)
            self.log_dict({f"train_{k}": v for k, v in losses.items()},
                          batch_size=batch.phoneme_ids.size(0))
            return losses["loss"]

        # manual optimization: LSGAN critic on random mel chunks (nipponjo)
        # — during the GAN refinement phase the aligner is assumed converged,
        # so the binarization loss runs at full scale from step 0 (nipponjo)
        opt_g, opt_d = self.optimizers()
        losses, pred_spect = self._step(batch)
        gen_loss = losses["loss"]
        mel, mel_len = batch.mels, batch.mel_lengths

        tl = int(min(mel_len.min().item(), self.tar_len))
        ofx = (torch.rand(mel_len.size(), device=self.device) * (mel_len + tl / 2) - tl / 2) \
            .clamp(mel_len * 0, mel_len - tl - 1).long()
        org = (extract_chunks(mel, ofx, mel_ids=None, chunk_len=tl).unsqueeze(1) + 4.5) / 2.5
        gen = (extract_chunks(pred_spect.transpose(1, 2), ofx, mel_ids=None,
                              chunk_len=tl).unsqueeze(1) + 4.5) / 2.5
        # critic (LSGAN)
        d_org, fmaps_org = self.critic(org)
        d_gen, _ = self.critic(gen.detach())
        loss_d = 0.5 * (d_org - 1).square().mean() + 0.5 * d_gen.square().mean()
        opt_d.zero_grad()
        self.manual_backward(loss_d)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1000.0)
        opt_d.step()
        # generator adversarial + feature matching
        d_gen2, fmaps_gen = self.critic(gen)
        # adversarial weight 4.0 and generator grad-clip 20, per nipponjo
        loss_g_adv = 4.0 * 0.5 * (d_gen2 - 1).square().mean()
        loss_fm = calc_feature_match_loss(fmaps_gen, [f.detach() for f in fmaps_org])
        gen_loss = gen_loss + loss_g_adv + loss_fm

        opt_g.zero_grad()
        self.manual_backward(gen_loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 20.0)
        opt_g.step()
        self.log_dict({"train_loss": gen_loss, "train_loss_d": loss_d,
                       "train_mel_loss": losses["mel_loss"]},
                      prog_bar=True, batch_size=batch.phoneme_ids.size(0))
        return gen_loss

    def validation_step(self, batch: Batch, batch_idx: int):
        losses, _ = self._step(batch)
        self.log_dict({f"val_{k}": v for k, v in losses.items()},
                      batch_size=batch.phoneme_ids.size(0))
        return losses["loss"]

    def configure_optimizers(self):
        if self.train_gan:
            # GAN loop: zero-momentum betas, lr 1e-4 for both generator and
            # critic, no scheduler (nipponjo recipe)
            opt_g = torch.optim.AdamW(
                self.model.parameters(), lr=self.hparams.gan_learning_rate,
                betas=(0.0, 0.99), weight_decay=self.hparams.weight_decay)
            opt_d = torch.optim.AdamW(
                self.critic.parameters(), lr=self.hparams.gan_learning_rate,
                betas=(0.0, 0.99), weight_decay=self.hparams.weight_decay)
            return [opt_g, opt_d]
        # AdamW(lr=1e-3, wd=1e-6) + Noam warmup as in NeMo's NoamAnnealing
        # (betas (0.9, 0.98) follow nipponjo's non-GAN recipe)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.hparams.learning_rate,
            betas=(0.9, 0.98), weight_decay=self.hparams.weight_decay)
        warmup = max(1, self.hparams.warmup_steps)

        def noam(step: int) -> float:
            # NeMo NoamAnnealing with d_model=1: factor = min(step^-0.5,
            # step * warmup^-1.5), NOT renormalized to peak at lr — the
            # effective peak LR (at step == warmup) is lr / sqrt(warmup),
            # i.e. ~3.16e-5 for the 1e-3 / 1000-step defaults
            step = max(1, step)
            return min(step ** -0.5, step * warmup ** -1.5)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, noam)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
