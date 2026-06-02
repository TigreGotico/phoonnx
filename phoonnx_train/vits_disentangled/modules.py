import math

import torch
from torch import nn

from vits.modules import (
    ConvFlow,
    DDSConv,
    ElementwiseAffine,
    Flip,
    LayerNorm,
    Log,
    ResBlock1,
    ResBlock2,
    WN,
    ConvReluNorm,
    fused_add_tanh_sigmoid_multiply,
    get_padding,
    init_weights,
    piecewise_rational_quadratic_transform,
)

# Re-export unchanged symbols for convenience
__all__ = [
    "LayerNorm",
    "ConvReluNorm",
    "DDSConv",
    "WN",
    "ResBlock1",
    "ResBlock2",
    "Log",
    "Flip",
    "ElementwiseAffine",
    "ResidualCouplingLayer",
    "ResidualCouplingBlock",
    "ConvFlow",
]


class ResidualCouplingLayer(nn.Module):
    """ResidualCouplingLayer with optional multi-factor conditioning (g_factors)."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: float = 0,
        gin_channels: int = 0,
        num_g_factors: int = 1,
        mean_only: bool = False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.num_g_factors = num_g_factors

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(
            hidden_channels, self.half_channels * (2 - mean_only), 1
        )
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, g_factors=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class ResidualCouplingBlock(nn.Module):
    """ResidualCouplingBlock with optional multi-factor conditioning (g_factors)."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
        num_g_factors: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels
        self.num_g_factors = num_g_factors

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    num_g_factors=num_g_factors,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, g_factors=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, g_factors=g_factors, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, g_factors=g_factors, reverse=reverse)
        return x
