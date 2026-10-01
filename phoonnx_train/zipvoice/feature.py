"""Vocos-style log-mel features from ZipVoice ``zipvoice/utils/feature.py``,
without the lhotse ``FeatureExtractor`` plumbing (only the math)."""
import math
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
import torchaudio


@dataclass
class VocosFbankConfig:
    sampling_rate: int = 24000
    n_mels: int = 100
    n_fft: int = 1024
    hop_length: int = 256


class VocosFbank:
    """wav (at ``sampling_rate``) -> log-mel ``(time, n_mels)``, matching the
    features the ZipVoice checkpoints were trained on (power=1 magnitude
    mel, ``log(clamp(min=1e-7))``, frame count = round(samples/hop))."""

    def __init__(self, config: VocosFbankConfig = VocosFbankConfig()):
        self.config = config
        self.fbank = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sampling_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            center=True,
            power=1,
        )

    @property
    def frame_shift(self) -> float:
        return self.config.hop_length / self.config.sampling_rate

    def extract(
        self,
        samples: Union[np.ndarray, torch.Tensor],
        sampling_rate: int,
    ) -> torch.Tensor:
        assert sampling_rate == self.config.sampling_rate, (
            f"Mismatched sampling rate: extractor expects "
            f"{self.config.sampling_rate}, got {sampling_rate}"
        )
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
        if samples.dim() == 1:
            samples = samples.unsqueeze(0)
        if samples.shape[0] > 1:
            samples = samples.mean(dim=0, keepdim=True)

        mel = self.fbank(samples.float()).clamp(min=1e-7).log()
        mel = mel.reshape(-1, mel.shape[-1]).t()  # (time, n_mels)

        # lhotse compute_num_frames: round to the nearest frame of the
        # true duration
        num_frames = int(
            math.ceil(round(samples.shape[1] / self.config.hop_length, ndigits=8)))
        if mel.shape[0] > num_frames:
            mel = mel[:num_frames]
        elif mel.shape[0] < num_frames:
            mel = torch.nn.functional.pad(
                mel.unsqueeze(0), (0, 0, 0, num_frames - mel.shape[0]),
                mode="replicate").squeeze(0)
        return mel
