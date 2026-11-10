"""Loss functions for the speech-autoencoder GAN stage: multi-resolution mel
reconstruction, least-squares adversarial losses, and discriminator
feature-matching.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio

MEL_RESOLUTIONS = ((256, 32), (512, 64), (1024, 80))  # (n_fft, n_mels); hop = n_fft // 4


class MultiResolutionMelLoss(torch.nn.Module):
    def __init__(self, sample_rate: int, resolutions=MEL_RESOLUTIONS):
        super().__init__()
        self.transforms = torch.nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_fft=n_fft, win_length=n_fft,
                hop_length=n_fft // 4, n_mels=n_mels, power=1.0, center=True)
            for n_fft, n_mels in resolutions
        ])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        total = x.new_zeros(())
        for tf in self.transforms:
            mx = torch.log(tf(x).clamp_min(1e-5))
            my = torch.log(tf(y).clamp_min(1e-5))
            total = total + F.l1_loss(mx, my)
        return total / len(self.transforms)


def discriminator_loss(real_outs, fake_outs) -> torch.Tensor:
    total = 0.0
    for real, fake in zip(real_outs, fake_outs):
        total = total + ((real - 1) ** 2).mean() + ((fake + 1) ** 2).mean()
    return total / len(real_outs)


def generator_adv_loss(fake_outs) -> torch.Tensor:
    return sum(((fake - 1) ** 2).mean() for fake in fake_outs) / len(fake_outs)


def feature_matching_loss(real_feats, fake_feats) -> torch.Tensor:
    total, n = 0.0, 0
    for real_layers, fake_layers in zip(real_feats, fake_feats):
        for r, f in zip(real_layers, fake_layers):
            total = total + F.l1_loss(f, r.detach())
            n += 1
    return total / max(n, 1)
