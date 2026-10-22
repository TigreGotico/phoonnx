"""GAN discriminators for the speech autoencoder: a HiFi-GAN-style multi-period
discriminator over the raw waveform and a multi-resolution discriminator over
log-magnitude spectrograms. Each returns per-sub-discriminator scores plus the
intermediate feature maps used by the feature-matching loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

PERIODS = (2, 3, 5, 7, 11)
PERIOD_CHANNELS = (16, 64, 128, 256, 256)
FFT_SIZES = (512, 1024, 2048)


class _PeriodDisc(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        chans = (1,) + PERIOD_CHANNELS
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(chans[i], chans[i + 1], (5, 1),
                                  (3, 1) if i < len(chans) - 2 else (1, 1), padding=(2, 0)))
            for i in range(len(chans) - 1)
        ])
        self.out = weight_norm(nn.Conv2d(PERIOD_CHANNELS[-1], 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor):
        b, t = x.shape
        if t % self.period:
            x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        x = x.view(b, 1, -1, self.period)
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feats.append(x)
        x = self.out(x)
        feats.append(x)
        return x.flatten(1), feats


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=PERIODS):
        super().__init__()
        self.discs = nn.ModuleList([_PeriodDisc(p) for p in periods])

    def forward(self, x: torch.Tensor):
        outs, feats = [], []
        for disc in self.discs:
            o, f = disc(x)
            outs.append(o)
            feats.append(f)
        return outs, feats


class _ResolutionDisc(nn.Module):
    def __init__(self, n_fft: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop = n_fft // 4
        self.win = n_fft
        specs = [(1, 16, (2, 1)), (16, 16, (2, 1)), (16, 16, (2, 1)), (16, 16, (1, 1))]
        self.convs = nn.ModuleList(
            [weight_norm(nn.Conv2d(ci, co, (5, 5), stride=s, padding=(2, 2))) for ci, co, s in specs]
        )
        self.out = weight_norm(nn.Conv2d(16, 1, (3, 3), 1, padding=(1, 1)))

    def _spec(self, x: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(self.win, device=x.device)
        spec = torch.stft(x, self.n_fft, self.hop, self.win, window=window, center=True, return_complex=True)
        return torch.log(spec.abs().clamp_min(1e-5))

    def forward(self, x: torch.Tensor):
        x = self._spec(x).unsqueeze(1)
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feats.append(x)
        x = self.out(x)
        feats.append(x)
        return x.flatten(1), feats


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, fft_sizes=FFT_SIZES):
        super().__init__()
        self.discs = nn.ModuleList([_ResolutionDisc(n) for n in fft_sizes])

    def forward(self, x: torch.Tensor):
        outs, feats = [], []
        for disc in self.discs:
            o, f = disc(x)
            outs.append(o)
            feats.append(f)
        return outs, feats
