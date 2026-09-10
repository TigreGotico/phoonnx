"""
GlowTTS text encoder.

Reconstructed from the GlowTTS paper (Kim et al. 2020, §3.1): a phoneme
embedding, a conv "prenet" (``ConvReluNorm``), a Transformer encoder stack
(shared with VITS — ``phoonnx_train.vits.attentions.Encoder``, reused
verbatim since VITS's text encoder descends from the same GlowTTS design),
and a final projection to per-token Gaussian prior statistics
(``m``, ``logs``) plus a duration predictor branch.
"""
import math

import torch
from torch import nn

from phoonnx_train.vits import commons
from phoonnx_train.vits.attentions import Encoder as TransformerEncoder
from phoonnx_train.vits.modules import ConvReluNorm

from phoonnx_train.glowtts.duration_predictor import DurationPredictor


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        filter_channels_dp: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        prenet_n_layers: int = 3,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.gin_channels = gin_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.prenet = ConvReluNorm(
            hidden_channels, hidden_channels, hidden_channels,
            kernel_size=5, n_layers=prenet_n_layers, p_dropout=0.5,
        )

        self.encoder = TransformerEncoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout,
        )

        self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj_s = nn.Conv1d(hidden_channels, out_channels, 1)

        self.duration_predictor = DurationPredictor(
            hidden_channels, filter_channels_dp, kernel_size=3, p_dropout=p_dropout,
            gin_channels=gin_channels,
        )

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor, g: torch.Tensor = None):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).type_as(x)

        x = self.prenet(x, x_mask)
        x = self.encoder(x * x_mask, x_mask)

        m = self.proj_m(x) * x_mask
        logs = self.proj_s(x) * x_mask

        logw = self.duration_predictor(x, x_mask, g=g)
        return x, m, logs, logw, x_mask
