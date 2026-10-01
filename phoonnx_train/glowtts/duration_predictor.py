"""
GlowTTS duration predictor.

GlowTTS (unlike VITS's stochastic duration predictor) uses a small,
deterministic conv-stack duration predictor trained with an MSE loss against
log-durations derived from Monotonic Alignment Search (see
:mod:`phoonnx_train.glowtts.monotonic_align`). Architecture reconstructed
from the GlowTTS paper (Kim et al. 2020, §3.2) — it mirrors the
``DurationPredictor`` used by VITS (``phoonnx_train/vits/models.py``), which
itself descends from the same GlowTTS design; the two are re-derived here
independently (not imported) so this package stays self-contained.
"""
import torch
from torch import nn

from phoonnx_train.vits.modules import LayerNorm


class DurationPredictor(nn.Module):
    """Two-layer conv duration predictor -> scalar log-duration per token."""

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor = None) -> torch.Tensor:
        x = torch.detach(x)
        if g is not None:
            x = x + self.cond(torch.detach(g))
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask
