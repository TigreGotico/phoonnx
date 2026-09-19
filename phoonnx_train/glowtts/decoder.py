"""
GlowTTS invertible flow decoder.

Reconstructed from the GlowTTS paper (Kim et al. 2020, §3.1, "Decoder"): a
stack of flow blocks, each block = squeeze -> [ActNorm -> InvertibleConv1x1
-> AffineCoupling] x n_blocks -> unsqueeze. This is the same flow family
Glow (Kingma & Dhariwal 2018) and WaveGlow use. The affine-coupling
conditioner network reuses ``phoonnx_train.vits.modules.WN`` (a WaveNet-style
gated-conv stack) verbatim — VITS's own residual coupling flow already reuses
this exact building block for the same purpose.

Every module below implements both a forward (mel -> latent, used in
training to compute the exact log-likelihood via the change-of-variables
formula) and a reverse pass (latent -> mel, used at inference/export time).
"""
from typing import Optional, Tuple

import torch
from torch import nn

from phoonnx_train.vits.modules import WN


class ActNorm(nn.Module):
    """Per-channel affine activation normalization with data-dependent init."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.logs = nn.Parameter(torch.zeros(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))
        self.register_buffer("initialized", torch.tensor(False))

    def initialize(self, x: torch.Tensor, x_mask: torch.Tensor) -> None:
        with torch.no_grad():
            denom = x_mask.sum([0, 2]).clamp_min(1.0)
            m = (x * x_mask).sum([0, 2]) / denom
            m_sq = ((x * x_mask) ** 2).sum([0, 2]) / denom
            v = m_sq - m**2
            logs = 0.5 * torch.log(v.clamp_min(1e-6))
            self.bias.data.copy_((-m * torch.exp(-logs)).view(1, self.channels, 1))
            self.logs.data.copy_((-logs).view(1, self.channels, 1))
            self.initialized.fill_(True)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, reverse: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not reverse and not bool(self.initialized) and self.training:
            self.initialize(x, x_mask)

        if reverse:
            z = (x - self.bias) * torch.exp(-self.logs) * x_mask
            return z, None
        z = (self.bias + torch.exp(self.logs) * x) * x_mask
        logdet = torch.sum(self.logs) * torch.sum(x_mask, [1, 2])
        return z, logdet


class InvConvNear(nn.Module):
    """Invertible 1x1 conv, computed on groups of ``n_split`` channels."""

    def __init__(self, channels: int, n_split: int = 4):
        super().__init__()
        assert channels % n_split == 0
        self.channels = channels
        self.n_split = n_split
        w_init = torch.linalg.qr(torch.randn(n_split, n_split))[0]
        if torch.det(w_init) < 0:
            w_init[:, 0] = -w_init[:, 0]
        self.weight = nn.Parameter(w_init)
        # Populated by store_inverse() before ONNX export: torch.inverse has
        # no ONNX symbolic (aten::linalg_inv is unsupported at any opset), so
        # the reverse (inference) pass must consume a precomputed inverse
        # weight instead of tracing the matrix inversion itself.
        self.weight_inv: Optional[torch.Tensor] = None

    def store_inverse(self) -> None:
        """Precompute and cache the inverse weight for a trace-safe reverse pass."""
        with torch.no_grad():
            self.weight_inv = torch.inverse(self.weight.double()).float().detach()

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, reverse: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, c, t = x.size()
        assert c % self.n_split == 0
        x_len = torch.sum(x_mask, [1, 2])

        x = x.view(b, 2, c // self.n_split, self.n_split // 2, t)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(b, self.n_split, c // self.n_split, t)

        if reverse:
            weight = self.weight_inv if self.weight_inv is not None else torch.inverse(self.weight.double()).float()
            logdet = None
        else:
            weight = self.weight
            logdet = torch.logdet(self.weight) * (c / self.n_split) * x_len

        weight = weight.view(self.n_split, self.n_split, 1, 1)
        z = torch.nn.functional.conv2d(x, weight)

        z = z.view(b, 2, self.n_split // 2, c // self.n_split, t)
        z = z.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * x_mask
        return z, logdet


class AffineCouplingLayer(nn.Module):
    """WN-conditioned affine coupling: splits channels, transforms one half."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.channels = channels
        self.half_channels = channels // 2

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.post = nn.Conv1d(hidden_channels, self.half_channels * 2, 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x0, x1 = torch.split(x, [self.half_channels] * 2, dim=1)

        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        m, logs = torch.split(stats, [self.half_channels] * 2, dim=1)

        if reverse:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            logdet = None
        else:
            x1 = (m + torch.exp(logs) * x1) * x_mask
            logdet = torch.sum(logs * x_mask, [1, 2])

        z = torch.cat([x0, x1], dim=1)
        return z, logdet


class FlowBlock(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, kernel_size: int,
                 dilation_rate: int, n_layers: int, gin_channels: int = 0):
        super().__init__()
        self.actnorm = ActNorm(channels)
        self.invconv = InvConvNear(channels, n_split=4)
        self.coupling = AffineCouplingLayer(
            channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels
        )

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            x, ld1 = self.actnorm(x, x_mask, reverse=False)
            x, ld2 = self.invconv(x, x_mask, reverse=False)
            x, ld3 = self.coupling(x, x_mask, g=g, reverse=False)
            return x, ld1 + ld2 + ld3
        x, _ = self.coupling(x, x_mask, g=g, reverse=True)
        x, _ = self.invconv(x, x_mask, reverse=True)
        x, _ = self.actnorm(x, x_mask, reverse=True)
        return x, None

    def store_inverse(self) -> None:
        self.invconv.store_inverse()


def _squeeze(x: torch.Tensor, x_mask: torch.Tensor, n_sqz: int = 2):
    b, c, t = x.size()
    t = (t // n_sqz) * n_sqz
    x = x[:, :, :t].view(b, c, t // n_sqz, n_sqz).permute(0, 3, 1, 2).contiguous().view(b, c * n_sqz, t // n_sqz)
    x_mask = x_mask[:, :, n_sqz - 1::n_sqz]
    return x * x_mask, x_mask


def _unsqueeze(x: torch.Tensor, x_mask: torch.Tensor, n_sqz: int = 2):
    b, c, t = x.size()
    x = x.view(b, n_sqz, c // n_sqz, t).permute(0, 2, 3, 1).contiguous().view(b, c // n_sqz, t * n_sqz)
    x_mask = x_mask.unsqueeze(-1).repeat(1, 1, 1, n_sqz).view(b, 1, t * n_sqz)
    return x * x_mask, x_mask


class FlowDecoder(nn.Module):
    """Squeeze -> n_blocks x FlowBlock -> unsqueeze, invertible mel<->latent."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_blocks: int,
        n_layers: int,
        n_sqz: int = 2,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_sqz = n_sqz
        sq_channels = in_channels * n_sqz
        self.blocks = nn.ModuleList([
            FlowBlock(sq_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor = None, reverse: bool = False):
        x, x_mask_sq = _squeeze(x, x_mask, self.n_sqz)
        logdet_total = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

        blocks = self.blocks if not reverse else reversed(list(self.blocks))
        for block in blocks:
            x, logdet = block(x, x_mask_sq, g=g, reverse=reverse)
            if logdet is not None:
                logdet_total = logdet_total + logdet

        x, _ = _unsqueeze(x, x_mask_sq, self.n_sqz)
        return x, (logdet_total if not reverse else None)

    def store_inverse(self) -> None:
        """Precompute cached inverse weights on every flow block's invertible 1x1
        conv, required before an ONNX trace of the reverse (inference) pass —
        see ``InvConvNear.store_inverse``."""
        for block in self.blocks:
            block.store_inverse()
