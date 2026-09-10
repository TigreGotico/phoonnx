"""Generator, discriminators and GAN losses for Vocos training.

The generator combines the vendored ConvNeXt backbone
(:mod:`phoonnx_train.vocos.backbone`) with the ISTFT head
(:mod:`phoonnx_train.vocos.head`).

Discriminators and losses follow the Vocos paper (Siuzdak, 2023):
a multi-period discriminator (as introduced by HiFi-GAN, Kong et al.,
2020) plus a multi-resolution *spectral* discriminator operating on STFT
magnitudes (as introduced by UnivNet, Jang et al., 2021), trained with the
hinge GAN objective, feature matching and an L1 mel-spectrogram loss.
"""
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
# the legacy weight_norm keeps the reference checkpoint key layout
# (weight_g / weight_v); the parametrized replacement renames the tensors
# and would silently break discriminator warm starts
from torch.nn.utils import weight_norm

from phoonnx_train.vocos.backbone import VocosBackbone
from phoonnx_train.vocos.head import ISTFTHead


class VocosGenerator(nn.Module):
    """Backbone + head with the head's pre-ISTFT coefficients exposed.

    ``forward`` returns audio ``[B, N]``.  ``coefficients`` returns the
    ``(mag, x, y)`` triple the phoonnx runtime Vocos adapter consumes,
    which is what gets exported to ONNX.  The head uses a centered ISTFT,
    matching the runtime adapter, which trims ``n_fft // 2`` samples from
    each side after overlap-add.
    """

    def __init__(
        self,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        dim: int = 512,
        intermediate_dim: int = 1536,
        num_layers: int = 8,
    ):
        super().__init__()
        self.backbone = VocosBackbone(
            input_channels=n_mels,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
        )
        self.head = ISTFTHead(dim=dim, n_fft=n_fft, hop_length=hop_length)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(mel))

    def coefficients(
        self, mel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """STFT coefficients ``(mag, x, y)``, each ``[B, n_fft//2+1, T]``."""
        return self.head.coefficients(self.backbone(mel))


# ----------------------------------------------------------------------
# Discriminators
# ----------------------------------------------------------------------

class _PeriodDiscriminator(nn.Module):
    """Single-period sub-discriminator (HiFi-GAN, Kong et al., 2020)."""

    def __init__(self, period: int):
        super().__init__()
        self.period = period
        chans = [1, 32, 128, 512, 1024]
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv2d(chans[i], chans[i + 1], (5, 1), (3, 1), padding=(2, 0))
                )
                for i in range(4)
            ]
            + [weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0)))]
        )
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        b, t = x.shape
        if t % self.period:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), mode="reflect")
            t = t + pad
        x = x.view(b, 1, t // self.period, self.period)
        fmaps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmaps.append(x)
        x = self.conv_post(x)
        fmaps.append(x)
        return x.flatten(1, -1), fmaps


class MultiPeriodDiscriminator(nn.Module):
    """Multi-period discriminator over prime periods (HiFi-GAN)."""

    def __init__(self, periods: Tuple[int, ...] = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList(
            _PeriodDiscriminator(p) for p in periods
        )

    def forward(self, x: torch.Tensor):
        scores, fmaps = [], []
        for d in self.discriminators:
            s, f = d(x)
            scores.append(s)
            fmaps.append(f)
        return scores, fmaps


class _ResolutionDiscriminator(nn.Module):
    """Single-resolution spectral sub-discriminator (UnivNet, Jang et al., 2021)."""

    def __init__(self, resolution: Tuple[int, int, int]):
        super().__init__()
        self.resolution = resolution  # (n_fft, hop_length, win_length)
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
            ]
        )
        self.conv_post = weight_norm(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def _spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolution
        spec = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length, device=x.device),
            center=True,
            return_complex=True,
        )
        return spec.abs().unsqueeze(1)  # [B, 1, F, T]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self._spectrogram(x)
        fmaps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmaps.append(x)
        x = self.conv_post(x)
        fmaps.append(x)
        return x.flatten(1, -1), fmaps


class MultiResolutionDiscriminator(nn.Module):
    """Multi-resolution spectral discriminator (UnivNet resolutions)."""

    def __init__(
        self,
        resolutions: Tuple[Tuple[int, int, int], ...] = (
            (1024, 256, 1024),
            (2048, 512, 2048),
            (512, 128, 512),
        ),
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            _ResolutionDiscriminator(r) for r in resolutions
        )

    def forward(self, x: torch.Tensor):
        scores, fmaps = [], []
        for d in self.discriminators:
            s, f = d(x)
            scores.append(s)
            fmaps.append(f)
        return scores, fmaps


# ----------------------------------------------------------------------
# Losses (hinge GAN objective, as in the Vocos paper)
# ----------------------------------------------------------------------

def discriminator_loss(
    real_scores: List[torch.Tensor], fake_scores: List[torch.Tensor]
) -> torch.Tensor:
    """Hinge loss for the discriminators."""
    loss = 0.0
    for real, fake in zip(real_scores, fake_scores):
        loss = loss + torch.mean(F.relu(1 - real)) + torch.mean(F.relu(1 + fake))
    return loss


def generator_adversarial_loss(fake_scores: List[torch.Tensor]) -> torch.Tensor:
    """Hinge adversarial loss for the generator."""
    loss = 0.0
    for fake in fake_scores:
        loss = loss + torch.mean(F.relu(1 - fake))
    return loss


def feature_matching_loss(
    real_fmaps: List[List[torch.Tensor]], fake_fmaps: List[List[torch.Tensor]]
) -> torch.Tensor:
    """L1 distance between real/fake discriminator feature maps."""
    loss = 0.0
    for real_layers, fake_layers in zip(real_fmaps, fake_fmaps):
        for real, fake in zip(real_layers, fake_layers):
            loss = loss + F.l1_loss(fake, real.detach())
    return loss
