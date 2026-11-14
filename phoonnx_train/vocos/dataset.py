"""Audio-only dataset for Vocos vocoder training.

Each item is a random fixed-length waveform crop, resampled to the target
sample rate.  Mel spectrograms are computed on-device in the Lightning
module with the exact acoustic-model mel settings
(:func:`phoonnx_train.vits.mel_processing.mel_spectrogram_torch`).
"""
import logging
from pathlib import Path
from typing import List, Sequence

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from phoonnx_train.vocos.data import MelConfig

_LOGGER = logging.getLogger(__name__)


class AudioCropDataset(Dataset):
    """Random fixed-length crops from a list of audio files."""

    def __init__(
        self,
        files: Sequence[Path],
        mel: MelConfig,
        crop_samples: int,
        train: bool = True,
    ):
        if not files:
            raise ValueError("no audio files to train on")
        self.files: List[Path] = list(files)
        self.mel = mel
        self.crop_samples = crop_samples
        self.train = train

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.files[index]
        try:
            audio, _sr = librosa.load(path, sr=self.mel.sample_rate, mono=True)
        except Exception:
            _LOGGER.warning("Unreadable audio file, substituting silence: %s", path)
            audio = np.zeros(self.crop_samples, dtype=np.float32)

        if len(audio) < self.crop_samples:
            audio = np.pad(audio, (0, self.crop_samples - len(audio)))
        elif len(audio) > self.crop_samples:
            if self.train:
                start = np.random.randint(0, len(audio) - self.crop_samples + 1)
            else:
                start = (len(audio) - self.crop_samples) // 2
            audio = audio[start:start + self.crop_samples]

        audio = np.clip(audio, -1.0, 1.0)
        return torch.from_numpy(audio.astype(np.float32))
