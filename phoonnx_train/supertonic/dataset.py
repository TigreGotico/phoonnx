"""Filelist datasets for the three SuperTonic stages.

SuperTonic is grapheme-level and G2P-free, so it consumes raw text rather than
the phoneme ids in phoonnx's shared ``dataset.jsonl``. The filelist is one
utterance per line::

    relative/path/to/audio.wav|transcript text|lang_code

``lang_code`` is optional and defaults to ``en``. Paths resolve against
``root_dir`` and any ``soundfile``-readable audio format works.

* :class:`WaveformDataset` (stage 1) returns fixed-length random audio crops for
  the GAN-trained autoencoder.
* :class:`TextAudioDataset` (stages 2 and 3) returns ``(waveform, text_ids)``;
  target latents are produced on the fly by a frozen autoencoder in the training
  loop, so this dataset only loads audio and tokenizes text.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from phoonnx_train.supertonic.text import CharTokenizer


def load_audio(path: str) -> Tuple[torch.Tensor, int]:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(wav).mean(dim=1), sr


def load_filelist(path: str) -> List[Tuple[str, str, str]]:
    entries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 2 or not parts[0] or not parts[1].strip():
            continue
        lang = parts[2].strip() if len(parts) > 2 and parts[2].strip() else "en"
        entries.append((parts[0], parts[1], lang))
    return entries


class WaveformDataset(Dataset):
    def __init__(self, filelist: str, root_dir: str, sample_rate: int, segment_samples: int):
        self.entries = load_filelist(filelist)
        self.root = Path(root_dir)
        self.sample_rate = sample_rate
        self.segment = segment_samples

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int) -> torch.Tensor:
        wav, sr = load_audio(str(self.root / self.entries[idx][0]))
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        n = self.segment
        if wav.shape[0] < n:
            return torch.nn.functional.pad(wav, (0, n - wav.shape[0]))
        start = random.randint(0, wav.shape[0] - n)
        return wav[start:start + n]


def waveform_collate(batch: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch, dim=0)


class TextAudioDataset(Dataset):
    def __init__(self, filelist: str, root_dir: str, tokenizer: CharTokenizer,
                 sample_rate: int, max_seconds: float = 10.0):
        self.entries = load_filelist(filelist)
        self.root = Path(root_dir)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_samples = int(max_seconds * sample_rate)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        wav_path, text, lang = self.entries[idx]
        wav, sr = load_audio(str(self.root / wav_path))
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.shape[0] > self.max_samples:
            wav = wav[:self.max_samples]
        ids = torch.tensor(self.tokenizer.encode(text, lang), dtype=torch.long)
        return wav, ids


def text_audio_collate(batch):
    wavs, ids = zip(*batch)
    wav_lens = torch.tensor([w.shape[0] for w in wavs], dtype=torch.long)
    text_lens = torch.tensor([t.shape[0] for t in ids], dtype=torch.long)
    wav_pad = torch.zeros(len(wavs), int(wav_lens.max()))
    for i, w in enumerate(wavs):
        wav_pad[i, :w.shape[0]] = w
    text_pad = torch.zeros(len(ids), int(text_lens.max().clamp_min(1)), dtype=torch.long)
    for i, t in enumerate(ids):
        text_pad[i, :t.shape[0]] = t
    return wav_pad, wav_lens, text_pad, text_lens
