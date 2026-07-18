"""LightningModule + dataset/collate wrapping the vendored ForwardTTS model.

Reuses the shared VITS preprocessing pipeline: ``audio_norm_path`` /
``audio_spec_path`` (linear spectrogram) are produced by
``phoonnx_train.preprocess``; the mel target is derived from the linear
spectrogram at load time via ``phoonnx_train.vits.mel_processing`` (no
separate mel cache needed). Pitch (F0) is read from the optional
``<utterance>.f0.npy`` sidecar caches written by
``ForwardTTSTrainingEngine.extra_preprocess``.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import LongTensor
from torch.utils.data import DataLoader, Dataset, random_split

from phoonnx_train.fastpitch.losses import ForwardTTSLoss
from phoonnx_train.fastpitch.model import ForwardTTS, ForwardTTSArgs
from phoonnx_train.fastpitch.pitch_stats import (
    f0_cache_path,
    load_or_compute_pitch_stats,
)
from phoonnx_train.vits.dataset import PhoonnxDataset, Utterance
from phoonnx_train.vits.mel_processing import spec_to_mel_torch

_LOG = logging.getLogger(__name__)

# SpeedySpeech-specific overrides layered on top of the tier preset when
# ``variant == "speedyspeech"``.
_SPEEDYSPEECH_OVERRIDES: Dict[str, Any] = {
    "encoder_type": "residual_conv_bn",
    "decoder_type": "residual_conv_bn",
    "use_pitch": False,
}


class UtteranceTensors:
    """Per-utterance training tensors: ids, mel (via spec), optional f0."""

    __slots__ = ("phoneme_ids", "spectrogram", "speaker_id", "pitch")

    def __init__(self, phoneme_ids: LongTensor, spectrogram: torch.Tensor,
                 speaker_id: Optional[LongTensor], pitch: Optional[torch.Tensor]):
        self.phoneme_ids = phoneme_ids
        self.spectrogram = spectrogram
        self.speaker_id = speaker_id
        self.pitch = pitch


class Batch:
    __slots__ = (
        "phoneme_ids", "phoneme_lengths", "mels", "mel_lengths",
        "pitch", "speaker_ids",
    )

    def __init__(self, phoneme_ids, phoneme_lengths, mels, mel_lengths, pitch, speaker_ids):
        self.phoneme_ids = phoneme_ids
        self.phoneme_lengths = phoneme_lengths
        self.mels = mels
        self.mel_lengths = mel_lengths
        self.pitch = pitch
        self.speaker_ids = speaker_ids


class ForwardTTSDataset(Dataset):
    """
    Wraps :class:`phoonnx_train.vits.dataset.PhoonnxDataset` utterances,
    converting the shared linear-spectrogram cache to mel at load time and
    optionally loading a pitch (F0) cache produced by
    ``ForwardTTSTrainingEngine.extra_preprocess``.
    """

    def __init__(self, dataset_paths: List[Path], mel_channels: int,
                 filter_length: int, sample_rate: int,
                 mel_fmin: float, mel_fmax: Optional[float],
                 max_phoneme_ids: Optional[int] = None):
        # accept either a preprocessed dataset dir or the dataset.jsonl itself
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
        self.pitch_mean, self.pitch_std = load_or_compute_pitch_stats(
            dataset_paths,
            [f0_cache_path(utt.audio_spec_path) for utt in self._inner.utterances],
        )

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> UtteranceTensors:
        utt: Utterance = self._inner.utterances[idx]
        spec = torch.load(utt.audio_spec_path)  # [n_fft//2+1, T]
        mel = spec_to_mel_torch(
            spec.unsqueeze(0), self.filter_length, self.mel_channels,
            self.sample_rate, self.mel_fmin, self.mel_fmax,
        ).squeeze(0)  # [mel_channels, T]

        pitch = None
        # optional sidecar cache written by extra_preprocess: "<stem>.f0.npy"
        f0_candidate = f0_cache_path(utt.audio_spec_path)
        if f0_candidate.exists():
            import numpy as np

            f0 = np.load(f0_candidate).astype("float32")
            # z-score voiced frames with the corpus stats; unvoiced stays 0
            voiced = f0 > 0
            f0[voiced] = (f0[voiced] - self.pitch_mean) / self.pitch_std
            pitch = torch.from_numpy(f0).unsqueeze(0)  # [1, T_f0]

        return UtteranceTensors(
            phoneme_ids=LongTensor(utt.phoneme_ids),
            spectrogram=mel,
            speaker_id=LongTensor([utt.speaker_id]) if utt.speaker_id is not None else None,
            pitch=pitch,
        )


class ForwardTTSCollate:
    def __init__(self, is_multispeaker: bool, use_pitch: bool):
        self.is_multispeaker = is_multispeaker
        self.use_pitch = use_pitch

    def __call__(self, utterances: List[UtteranceTensors]) -> Batch:
        n = len(utterances)
        max_ph = max(u.phoneme_ids.size(0) for u in utterances)
        max_mel = max(u.spectrogram.size(1) for u in utterances)
        mel_channels = utterances[0].spectrogram.size(0)

        phoneme_ids = torch.zeros(n, max_ph, dtype=torch.long)
        phoneme_lengths = torch.zeros(n, dtype=torch.long)
        mels = torch.zeros(n, mel_channels, max_mel)
        mel_lengths = torch.zeros(n, dtype=torch.long)
        pitch = torch.zeros(n, 1, max_mel) if self.use_pitch else None
        speaker_ids = torch.zeros(n, dtype=torch.long) if self.is_multispeaker else None

        for i, utt in enumerate(utterances):
            pl_ = utt.phoneme_ids.size(0)
            ml = utt.spectrogram.size(1)
            phoneme_ids[i, :pl_] = utt.phoneme_ids
            phoneme_lengths[i] = pl_
            mels[i, :, :ml] = utt.spectrogram
            mel_lengths[i] = ml
            if self.use_pitch and utt.pitch is not None:
                pl_len = min(utt.pitch.size(-1), max_mel)
                pitch[i, :, :pl_len] = utt.pitch[:, :pl_len]
            if speaker_ids is not None and utt.speaker_id is not None:
                speaker_ids[i] = utt.speaker_id

        return Batch(phoneme_ids, phoneme_lengths, mels, mel_lengths, pitch, speaker_ids)


class ForwardTTSModule(pl.LightningModule):
    """LightningModule wrapping the vendored :class:`ForwardTTS` model."""

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
        variant: str = "fastpitch",
        hidden_channels: int = 384,
        hidden_channels_ffn: int = 1024,
        encoder_num_layers: int = 6,
        decoder_num_layers: int = 6,
        num_heads: int = 1,
        use_pitch: Optional[bool] = None,
        dataset: Optional[List[Path]] = None,
        learning_rate: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.998),
        eps: float = 1e-9,
        weight_decay: float = 1e-6,
        batch_size: int = 8,
        num_workers: int = 1,
        validation_split: float = 0.1,
        num_test_examples: int = 5,
        max_phoneme_ids: Optional[int] = None,
        # binarization is delayed: ramping the binary alignment loss in from
        # epoch 0 risks locking in a still-random soft alignment
        binary_loss_start_epoch: int = 10,
        binary_loss_warmup_epochs: int = 10,
        **kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters()

        variant_overrides: Dict[str, Any] = {}
        if variant == "speedyspeech":
            variant_overrides.update(_SPEEDYSPEECH_OVERRIDES)
        if use_pitch is not None:
            variant_overrides["use_pitch"] = use_pitch

        args = ForwardTTSArgs(
            num_chars=num_symbols,
            out_channels=mel_channels,
            hidden_channels=hidden_channels,
            hidden_channels_ffn=hidden_channels_ffn,
            encoder_num_layers=encoder_num_layers,
            decoder_num_layers=decoder_num_layers,
            num_heads=num_heads,
            num_speakers=num_speakers,
            **variant_overrides,
        )
        self.model_args = args
        self.model = ForwardTTS(args)
        self.criterion = ForwardTTSLoss()

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
        full_dataset = ForwardTTSDataset(
            self.hparams.dataset,
            mel_channels=self.hparams.mel_channels,
            filter_length=self.hparams.filter_length,
            sample_rate=self.hparams.sample_rate,
            mel_fmin=self.hparams.mel_fmin,
            mel_fmax=self.hparams.mel_fmax,
            max_phoneme_ids=max_phoneme_ids,
        )
        valid_size = max(0, int(len(full_dataset) * validation_split))
        test_size = min(num_test_examples, max(0, len(full_dataset) - valid_size))
        train_size = len(full_dataset) - valid_size - test_size
        self._train_dataset, self._test_dataset, self._val_dataset = random_split(
            full_dataset, [train_size, test_size, valid_size]
        )

    def _collate(self) -> ForwardTTSCollate:
        return ForwardTTSCollate(
            is_multispeaker=self.hparams.num_speakers > 1,
            use_pitch=bool(self.model_args.use_pitch),
        )

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
    def forward(self, x, pace: float = 1.0, pitch_mul: float = 1.0,
               pitch_add: float = 0.0, speaker=None):
        return self.model.inference(x, speaker=speaker, pace=pace,
                                    pitch_mul=pitch_mul, pitch_add=pitch_add)

    def _step(self, batch: Batch, binary_loss_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        outputs = self.model(
            x=batch.phoneme_ids,
            x_lengths=batch.phoneme_lengths,
            y=batch.mels,
            y_lengths=batch.mel_lengths,
            pitch=batch.pitch,
            speaker=batch.speaker_ids,
        )
        losses = self.criterion(
            decoder_output=outputs["model_outputs"],
            decoder_target=batch.mels.transpose(1, 2),
            decoder_output_lens=batch.mel_lengths,
            dur_output=outputs["durations_log"],
            dur_target=outputs["durations"],
            input_lens=batch.phoneme_lengths,
            pitch_output=outputs["pitch_avg_pred"],
            pitch_target=outputs["pitch_avg"],
            aligner_logprob=outputs["alignment_logprob"],
            alignment_hard=outputs["alignment_hard"],
            alignment_soft=outputs["alignment_soft"],
            binary_loss_weight=binary_loss_weight,
        )
        return losses

    def training_step(self, batch: Batch, batch_idx: int):
        # ramp the binary alignment loss in over binary_loss_warmup_epochs
        # starting at binary_loss_start_epoch — full-strength binarization
        # against a still-random soft alignment destabilizes the aligner
        warmup = max(1, self.hparams.binary_loss_warmup_epochs)
        binary_loss_weight = min(1.0, max(
            0.0, (self.current_epoch - self.hparams.binary_loss_start_epoch + 1) / warmup))
        losses = self._step(batch, binary_loss_weight=binary_loss_weight)
        self.log_dict({f"train_{k}": v for k, v in losses.items()}, prog_bar=False,
                      batch_size=batch.phoneme_ids.size(0))
        return losses["loss"]

    def validation_step(self, batch: Batch, batch_idx: int):
        losses = self._step(batch)
        self.log_dict({f"val_{k}": v for k, v in losses.items()},
                      batch_size=batch.phoneme_ids.size(0))
        return losses["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
            eps=self.hparams.eps,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999875)
        return [optimizer], [scheduler]
