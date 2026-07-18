"""pytorch_lightning wrapper around the vendored ZipVoice model.

Lightning port of the upstream ``train_zipvoice.py`` loop (Zhu et al.,
"ZipVoice: Fast and high-quality zero-shot text-to-speech with flow
matching", arXiv:2506.13053): per batch, sample ``t ~ U(0,1)`` and Gaussian
noise, mask a random 70-100% span of each utterance as the generation
target (the rest is the speech prompt the model in-fills from), drop the
text condition for ``condition_drop_ratio`` of items (classifier-free
guidance) and regress the straight-line vector field, with the loss
restricted to target and non-padded frames. Optimizer: ScaledAdam + Eden
with the Zipformer batch-count schedules driven from the global step.

Consumes the shared phoonnx ``dataset.jsonl`` (phoneme_ids + cached
normalized audio); audio is resampled to 24 kHz and turned into the 100-bin
Vocos log-mel features ZipVoice expects, cached beside the audio cache.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch

LOG = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

# upstream train_zipvoice.py scales features by 0.1 both at training time
# and in the runtime adapter (phoonnx.engines.zipvoice.ZipVoiceAdapter
# FEAT_SCALE) — the model is defined over scaled features
FEAT_SCALE = 0.1


class ZipVoiceDataset(torch.utils.data.Dataset):
    """Shared-``dataset.jsonl`` utterances → (token ids, Vocos fbank).

    Features are returned pre-scaled by :data:`FEAT_SCALE`, matching the
    upstream recipe and the runtime adapter; the on-disk cache stores the
    raw (unscaled) fbank.
    """

    def __init__(self, dataset_paths: List[Path], sample_rate: int = 24000,
                 source_sample_rate: Optional[int] = None,
                 max_phoneme_ids: Optional[int] = None,
                 cache_features: bool = True):
        from phoonnx_train.matcha.jsonl import load_dataset_lines
        from phoonnx_train.zipvoice.feature import VocosFbank, VocosFbankConfig

        self.utterances: List[Dict[str, Any]] = []
        for p in dataset_paths:
            p = Path(p)
            jsonl = p / "dataset.jsonl" if p.is_dir() else p
            self.utterances.extend(
                load_dataset_lines(jsonl, max_phoneme_ids=max_phoneme_ids))
        self.sample_rate = sample_rate
        self.source_sample_rate = source_sample_rate or sample_rate
        self.cache_features = cache_features
        self.fbank = VocosFbank(VocosFbankConfig(sampling_rate=sample_rate))

    def __len__(self) -> int:
        return len(self.utterances)

    def _features(self, utt: Dict[str, Any]) -> torch.Tensor:
        import numpy as np

        cache = Path(str(utt["audio_norm_path"])).with_suffix(
            f".zipvoice-fbank-{self.sample_rate}.npy")
        if self.cache_features and cache.is_file():
            return torch.from_numpy(np.load(cache)) * FEAT_SCALE
        audio = torch.load(utt["audio_norm_path"])  # normalized audio tensor
        if audio.dim() > 1:
            audio = audio.reshape(-1)
        if self.source_sample_rate != self.sample_rate:
            import torchaudio
            audio = torchaudio.functional.resample(
                audio, self.source_sample_rate, self.sample_rate)
        feats = self.fbank.extract(audio, self.sample_rate)  # (T, n_mels)
        if self.cache_features:
            try:
                np.save(cache, feats.numpy())
            except OSError:
                pass
        return feats * FEAT_SCALE

    def __getitem__(self, idx: int):
        utt = self.utterances[idx]
        return list(utt["phoneme_ids"]), self._features(utt)


def collate_zipvoice(batch):
    tokens = [b[0] for b in batch]
    feats = [b[1] for b in batch]
    lens = torch.tensor([f.size(0) for f in feats], dtype=torch.int64)
    max_len = int(lens.max())
    dim = feats[0].size(1)
    features = torch.zeros(len(batch), max_len, dim)
    for i, f in enumerate(feats):
        features[i, :f.size(0)] = f
    return tokens, features, lens


# ----------------------------------------------------------------------
# LightningModule
# ----------------------------------------------------------------------

class ZipVoiceModule(pl.LightningModule):
    """Lightning port of upstream ``train_zipvoice.py``'s loop."""

    def __init__(self, num_symbols: int, sample_rate: int = 24000,
                 source_sample_rate: Optional[int] = None,
                 dataset: Optional[List[str]] = None,
                 batch_size: int = 8, num_workers: int = 2,
                 validation_split: float = 0.05,
                 max_phoneme_ids: Optional[int] = None,
                 model_params: Optional[Dict[str, Any]] = None,
                 base_lr: float = 0.02,
                 lr_batches: float = 7500,
                 lr_epochs: float = 10,
                 condition_drop_ratio: float = 0.2,
                 clipping_scale: float = 2.0,
                 **_ignored: Any):
        super().__init__()
        from phoonnx_train.zipvoice.model import ZipVoice

        self.save_hyperparameters()
        params = dict(model_params or {})
        params.setdefault("vocab_size", num_symbols)
        self.model = ZipVoice(**params)
        self._dataset_paths = [Path(p) for p in (dataset or [])]
        self._train_set = None
        self._val_set = None

    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None):
        if self._train_set is not None or not self._dataset_paths:
            return
        full = ZipVoiceDataset(
            self._dataset_paths, sample_rate=self.hparams.sample_rate,
            source_sample_rate=self.hparams.source_sample_rate,
            max_phoneme_ids=self.hparams.max_phoneme_ids)
        n_val = max(1, int(len(full) * self.hparams.validation_split)) \
            if len(full) > 1 else 0
        n_train = len(full) - n_val
        gen = torch.Generator().manual_seed(1234)
        if n_val:
            self._train_set, self._val_set = torch.utils.data.random_split(
                full, [n_train, n_val], generator=gen)
        else:
            self._train_set, self._val_set = full, None

    def _loader(self, ds, shuffle: bool):
        kwargs: Dict[str, Any] = dict(
            batch_size=self.hparams.batch_size, shuffle=shuffle,
            drop_last=shuffle, collate_fn=collate_zipvoice,
            num_workers=self.hparams.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.hparams.num_workers > 0)
        if self.hparams.num_workers > 0:
            kwargs["prefetch_factor"] = 4
        return torch.utils.data.DataLoader(ds, **kwargs)

    def train_dataloader(self):
        return self._loader(self._train_set, shuffle=True)

    def val_dataloader(self):
        if self._val_set is None:
            return None
        return self._loader(self._val_set, shuffle=False)

    # ------------------------------------------------------------------
    def _fm_loss(self, batch, is_training: bool) -> torch.Tensor:
        tokens, features, features_lens = batch
        b = features.size(0)
        noise = torch.randn_like(features)
        if is_training:
            t = torch.rand(b, 1, 1, device=features.device)
        else:
            # deterministic time grid for stable validation numbers
            t = (torch.arange(b, device=features.device) / max(b, 1)) \
                .view(b, 1, 1)
        return self.model(
            tokens=tokens, features=features, features_lens=features_lens,
            noise=noise, t=t,
            condition_drop_ratio=(self.hparams.condition_drop_ratio
                                  if is_training else 0.0))

    def training_step(self, batch, batch_idx):
        from phoonnx_train.zipvoice.common import set_batch_count
        set_batch_count(self.model, float(self.global_step))
        loss = self._fm_loss(batch, is_training=True)
        self.log("train/fm_loss", loss, prog_bar=True,
                 batch_size=batch[1].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._fm_loss(batch, is_training=False)
        self.log("val_loss", loss, prog_bar=True, batch_size=batch[1].size(0))

    def configure_optimizers(self):
        from phoonnx_train.zipvoice.lr_scheduler import Eden
        from phoonnx_train.zipvoice.optim import ScaledAdam

        params = [{"params": list(self.model.parameters()),
                   "names": [n for n, _ in self.model.named_parameters()]}]
        opt = ScaledAdam(params, lr=self.hparams.base_lr,
                         clipping_scale=self.hparams.clipping_scale)
        sched = Eden(opt, lr_batches=self.hparams.lr_batches,
                     lr_epochs=self.hparams.lr_epochs)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def on_train_epoch_start(self):
        sched = self.lr_schedulers()
        if sched is not None and hasattr(sched, "step_epoch"):
            sched.step_epoch(self.current_epoch)

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step_batch(self.global_step)


# ----------------------------------------------------------------------
# ONNX export wrappers (upstream onnx_export.py, verbatim math)
# ----------------------------------------------------------------------

class OnnxTextModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embed = model.embed
        self.text_encoder = model.text_encoder
        self.pad_id = model.pad_id

    def forward(self, tokens, prompt_tokens, prompt_features_len, speed):
        cat_tokens = torch.cat([prompt_tokens, tokens], dim=1)
        cat_tokens = torch.nn.functional.pad(cat_tokens, (0, 1),
                                             value=self.pad_id)
        tokens_len = cat_tokens.shape[1] - 1
        padding_mask = (torch.arange(tokens_len + 1) == tokens_len).unsqueeze(0)

        embed = self.embed(cat_tokens)
        embed = self.text_encoder(x=embed, t=None, padding_mask=padding_mask)

        features_len = torch.ceil(
            (prompt_features_len / prompt_tokens.shape[1] * tokens_len / speed)
        ).to(dtype=torch.int64)
        token_dur = torch.div(features_len, tokens_len,
                              rounding_mode="floor").to(dtype=torch.int64)
        # rank-0 (see upstream onnx_export.py: rank-1 scalars break ORT Concat)
        token_dur = token_dur.reshape(())
        features_len = features_len.reshape(())

        text_condition = embed[:, :-1, :].unsqueeze(2).expand(-1, -1, token_dur, -1)
        text_condition = text_condition.reshape(embed.shape[0], -1, embed.shape[2])
        text_condition = torch.cat(
            [text_condition,
             embed[:, -1:, :].expand(-1, features_len - text_condition.shape[1], -1)],
            dim=1)
        return text_condition


class OnnxFlowMatchingModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fm_decoder = model.fm_decoder
        self.model_func = model.forward_fm_decoder
        self.feat_dim = model.feat_dim

    def forward(self, t, x, text_condition, speech_condition, guidance_scale):
        x = x.repeat(2, 1, 1)
        text_condition = torch.cat(
            [torch.zeros_like(text_condition), text_condition], dim=0)
        speech_condition = torch.cat(
            [torch.where(t > 0.5, torch.zeros_like(speech_condition),
                         speech_condition),
             speech_condition], dim=0)
        guidance_scale = torch.where(t > 0.5, guidance_scale,
                                     guidance_scale * 2.0)
        data_uncond, data_cond = self.model_func(
            t=t, xt=x, text_condition=text_condition,
            speech_condition=speech_condition).chunk(2, dim=0)
        return (1 + guidance_scale) * data_cond - guidance_scale * data_uncond
