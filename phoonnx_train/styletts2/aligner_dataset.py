"""Datasets for the StyleTTS2 auxiliary-model trainers (text aligner, pitch
extractor).

Port of the yl4579/AuxiliaryASR and yl4579/PitchExtractor mel pipelines with
the same list format the StyleTTS2 engine uses (``filename|text|speaker`` +
``wavs/``), the shared 178-symbol ``TextCleaner`` table, plus two speed-ups:

- mel (and F0) features are computed once and cached as ``.npy`` next to the
  audio files;
- a length-bucketed batch sampler groups clips of similar length to cut
  padding waste.

The ``text`` field must already be phonemes (IPA, StyleTTS2 symbol set) — use
``phoonnx_train.styletts2.phonemize_corpus`` to phonemize raw-text lists.
"""
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Sampler

from phoonnx_train.styletts2.meldataset import MEL_PARAMS, SPECT_PARAMS, TextCleaner

LOG = logging.getLogger(__name__)

MEL_MEAN, MEL_STD = -4, 4


def parse_list_lines(lines: List[str], root_path: str) -> List[Tuple[str, str, int]]:
    """``filename|text|speaker`` lines -> (wav_path, phonemes, speaker_id)."""
    out = []
    for line in lines:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("|")
        if len(parts) == 2:
            parts.append("0")
        path, text, speaker = parts[0], parts[1], parts[2]
        if root_path and not Path(path).is_absolute():
            path = str(Path(root_path) / path)
        out.append((path, text, int(speaker) if speaker.strip().isdigit() else 0))
    return out


class AuxMelDataset(torch.utils.data.Dataset):
    """Mel + phoneme-id dataset for the aligner; optionally F0 for the pitch
    extractor (``with_f0=True`` computes pyworld harvest F0, cached)."""

    def __init__(self,
                 data_list: List[str],
                 root_path: str = "",
                 sr: int = 24000,
                 n_mels: int = MEL_PARAMS["n_mels"],
                 with_f0: bool = False,
                 cache_features: bool = True):
        self.data = parse_list_lines(data_list, root_path)
        self.sr = sr
        self.n_mels = n_mels
        self.with_f0 = with_f0
        self.cache_features = cache_features
        self.text_cleaner = TextCleaner()
        self.to_melspec = torchaudio.transforms.MelSpectrogram(
            n_mels=n_mels, **SPECT_PARAMS)
        # blank/silence token framing the utterance, as in AuxiliaryASR
        self.blank_index = self.text_cleaner.word_index_dictionary[" "]

    def __len__(self) -> int:
        return len(self.data)

    def _load_wave(self, wav_path: str) -> torch.Tensor:
        import soundfile as sf
        wave, sr = sf.read(wav_path, dtype="float32")
        wave = torch.from_numpy(wave)
        if wave.dim() > 1:
            wave = wave.mean(dim=-1)
        if sr != self.sr:
            wave = torchaudio.functional.resample(wave, sr, self.sr)
        return wave

    def _mel(self, wav_path: str) -> torch.Tensor:
        # cache key encodes the feature params so a stale cache from a
        # different sample rate / mel count is never silently reused
        cache = Path(wav_path).with_suffix(f".mel-{self.sr}-{self.n_mels}.npy")
        if self.cache_features and cache.is_file():
            return torch.from_numpy(np.load(cache))
        mel = self.to_melspec(self._load_wave(wav_path))
        if self.cache_features:
            try:
                np.save(cache, mel.numpy())
            except OSError:  # read-only dataset dir — just recompute next time
                pass
        return mel

    def _f0(self, wav_path: str, n_frames: int) -> torch.Tensor:
        cache = Path(wav_path).with_suffix(f".f0-{self.sr}.npy")
        if self.cache_features and cache.is_file():
            f0 = np.load(cache)
        else:
            import pyworld
            x = self._load_wave(wav_path).numpy().astype(np.float64)
            frame_period = SPECT_PARAMS["hop_length"] / self.sr * 1000
            f0, t = pyworld.harvest(x, self.sr, frame_period=frame_period)
            if np.count_nonzero(f0) < 5:  # harvest failed — retry with dio
                f0, t = pyworld.dio(x, self.sr, frame_period=frame_period)
            f0 = pyworld.stonemask(x, f0, t, self.sr)
            if self.cache_features:
                try:
                    np.save(cache, f0)
                except OSError:
                    pass
        f0 = torch.from_numpy(f0).float()
        # align to mel frame count
        if f0.numel() >= n_frames:
            f0 = f0[:n_frames]
        else:
            f0 = F.pad(f0, (0, n_frames - f0.numel()))
        return f0

    def mel_frames(self, idx: int) -> int:
        """Cheap-ish length probe for the bucket sampler (uses the cache)."""
        return self._mel(self.data[idx][0]).size(1)

    def __getitem__(self, idx: int):
        wav_path, phonemes, _speaker = self.data[idx]
        mel = self._mel(wav_path)
        text = self.text_cleaner(phonemes)
        text.insert(0, self.blank_index)
        text.append(self.blank_index)
        text = torch.LongTensor(text)

        # upstream AuxiliaryASR: stretch too-short mels so CTC stays valid.
        # Never stretch when F0 is requested — the F0 track is frame-aligned
        # to the *original* mel and stretching would desynchronize them.
        if not self.with_f0 and (text.size(0) + 1) >= (mel.size(1) // 3):
            mel = F.interpolate(mel.unsqueeze(0), size=(text.size(0) + 1) * 3,
                                align_corners=False, mode="linear").squeeze(0)
        mel = (torch.log(1e-5 + mel) - MEL_MEAN) / MEL_STD
        mel = mel[:, :mel.size(1) - mel.size(1) % 2]

        if self.with_f0:
            return mel, text, self._f0(wav_path, mel.size(1))
        return mel, text


class AlignerCollater:
    def __call__(self, batch):
        with_f0 = len(batch[0]) == 3
        batch = sorted(batch, key=lambda b: b[0].size(1), reverse=True)
        n_mels = batch[0][0].size(0)
        max_mel = max(b[0].size(1) for b in batch)
        max_text = max(b[1].size(0) for b in batch)
        bsz = len(batch)

        mels = torch.zeros(bsz, n_mels, max_mel)
        texts = torch.zeros(bsz, max_text, dtype=torch.long)
        text_lengths = torch.zeros(bsz, dtype=torch.long)
        mel_lengths = torch.zeros(bsz, dtype=torch.long)
        f0s = torch.zeros(bsz, max_mel) if with_f0 else None
        for i, item in enumerate(batch):
            mel, text = item[0], item[1]
            mels[i, :, :mel.size(1)] = mel
            texts[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)
            mel_lengths[i] = mel.size(1)
            if with_f0:
                f0s[i, :mel.size(1)] = item[2]
        if with_f0:
            return texts, text_lengths, mels, mel_lengths, f0s
        return texts, text_lengths, mels, mel_lengths


class LengthBucketSampler(Sampler):
    """Batches indices of similar mel length (less padding => faster steps),
    shuffling bucket contents and batch order every epoch."""

    def __init__(self, lengths: List[int], batch_size: int, shuffle: bool = True,
                 drop_last: bool = False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last and len(lengths) > batch_size
        self.order = np.argsort(lengths).tolist()

    def _batches(self):
        batches = [self.order[i:i + self.batch_size]
                   for i in range(0, len(self.order), self.batch_size)]
        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        return batches

    def __iter__(self):
        batches = self._batches()
        if self.shuffle:
            for b in batches:
                random.shuffle(b)
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return len(self._batches())


def build_aux_dataloader(dataset: AuxMelDataset,
                         batch_size: int,
                         num_workers: int,
                         validation: bool = False,
                         bucket_by_length: bool = True):
    kwargs: Dict = dict(num_workers=num_workers,
                        collate_fn=AlignerCollater(),
                        pin_memory=torch.cuda.is_available(),
                        persistent_workers=num_workers > 0)
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    if bucket_by_length and len(dataset) > batch_size:
        lengths = [dataset.mel_frames(i) for i in range(len(dataset))]
        kwargs["batch_sampler"] = LengthBucketSampler(
            lengths, batch_size, shuffle=not validation,
            drop_last=not validation)
    else:
        kwargs.update(batch_size=batch_size, shuffle=not validation,
                      drop_last=not validation)
    return torch.utils.data.DataLoader(dataset, **kwargs)
