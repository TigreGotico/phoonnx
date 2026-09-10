"""ConvNeXt backbone for the Vocos vocoder (Siuzdak, 2023).

Vocos keeps the temporal resolution constant end-to-end: a stack of
ConvNeXt blocks (Liu et al., 2022 — depthwise conv → LayerNorm →
pointwise MLP with GELU → layer scale → residual) maps the mel
spectrogram to hidden features, and the ISTFT head turns those into STFT
coefficients.  Parameter names follow the reference Vocos layout
(``embed``, ``norm``, ``convnext.N.*``, ``final_layer_norm``) so
published Vocos-family checkpoints warm-start with a near-identity key
mapping.
"""
from typing import Optional

import torch
from torch import nn


class ConvNeXtBlock(nn.Module):
    """One ConvNeXt block operating on ``[B, C, T]`` features."""

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = self.gamma * x
        x = x.transpose(1, 2)
        return residual + x


class VocosBackbone(nn.Module):
    """Mel ``[B, n_mels, T]`` → hidden features ``[B, T, dim]``."""

    def __init__(
        self,
        input_channels: int = 80,
        dim: int = 512,
        intermediate_dim: int = 1536,
        num_layers: int = 8,
        layer_scale_init_value: Optional[float] = None,
    ):
        super().__init__()
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList(
            ConvNeXtBlock(
                dim=dim,
                intermediate_dim=intermediate_dim,
                layer_scale_init_value=layer_scale_init_value or 1 / num_layers,
            )
            for _ in range(num_layers)
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            x = block(x)
        return self.final_layer_norm(x.transpose(1, 2))
