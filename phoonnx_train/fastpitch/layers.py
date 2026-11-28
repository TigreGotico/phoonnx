"""
Encoder/decoder/predictor layers for the vendored ForwardTTS.

Adapted from coqui-ai/TTS (© Coqui GmbH, Mozilla Public License 2.0):

- ``TTS/tts/layers/generic/transformer.py``   (FFTransformer blocks)
- ``TTS/tts/layers/generic/res_conv_bn.py``   (residual conv-BN blocks, SpeedySpeech)
- ``TTS/tts/layers/feed_forward/duration_predictor.py``

The multi-head attention is implemented manually (matmul + ``-1`` reshapes)
instead of ``torch.nn.MultiheadAttention`` so the traced ONNX graph stays
truly dynamic-length (see docs/fastpitch.md).
"""
import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Tracer-safe multi-head self-attention (batch_first)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C], mask: [B, 1, T] (1 = keep)."""
        b = x.shape[0]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # [B, H, T, Dh] with live-shape reshapes
        q = q.reshape(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        k = k.reshape(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = v.reshape(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn_mask = mask.unsqueeze(1)  # [B, 1, 1, T]
        scores = scores.masked_fill(attn_mask == 0, -1e4)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)  # [B, H, T, Dh]
        out = out.transpose(1, 2).reshape(b, -1, self.d_model)
        return self.out_proj(out)


class FFTransformerLayer(nn.Module):
    """One FFT block: self-attention + 1D-conv feed-forward (FastPitch style)."""

    def __init__(self, d_model: int, n_heads: int, d_ffn: int,
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, d_ffn, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(d_ffn, d_model, kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C], mask: [B, 1, T]."""
        x = self.norm1(x + self.dropout(self.attn(x, mask)))
        x = x * mask.transpose(1, 2)
        y = self.conv2(F.relu(self.conv1(x.transpose(1, 2)))).transpose(1, 2)
        x = self.norm2(x + self.dropout(y))
        return x * mask.transpose(1, 2)


class FFTransformerBlock(nn.Module):
    """Stack of FFT layers operating on [B, C, T] (channel-first, coqui convention)."""

    def __init__(self, in_out_channels: int, num_heads: int, hidden_channels_ffn: int,
                 num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [FFTransformerLayer(in_out_channels, num_heads, hidden_channels_ffn,
                                dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T], mask: [B, 1, T] -> [B, C, T]."""
        y = x.transpose(1, 2)
        for layer in self.layers:
            y = layer(y, mask)
        return y.transpose(1, 2)


class Conv1dBN(nn.Module):
    """Same-length conv+BN block. Handles even kernel sizes (e.g. the
    SpeedySpeech default ``kernel_size=4``) with asymmetric padding so the
    output time dimension always matches the input (needed since callers
    multiply by a fixed-length mask and add residuals)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        total_padding = dilation * (kernel_size - 1)
        self._pad = (total_padding // 2, total_padding - total_padding // 2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=0, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self._pad)
        return self.norm(F.relu(self.conv(x)))


class ResidualConv1dBNBlock(nn.Module):
    """Residual conv-BN stack (SpeedySpeech encoder/decoder building block)."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int,
                 kernel_size: int = 4, num_res_blocks: int = 13, dilations=None):
        super().__init__()
        dilations = dilations or [1] * num_res_blocks
        self.res_blocks = nn.ModuleList()
        for i, d in enumerate(dilations):
            in_c = in_channels if i == 0 else hidden_channels
            out_c = out_channels if i == len(dilations) - 1 else hidden_channels
            self.res_blocks.append(Conv1dBN(in_c, out_c, kernel_size, dilation=d))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T], mask: [B, 1, T]."""
        o = x * mask
        for i, block in enumerate(self.res_blocks):
            res = o
            o = block(o) * mask
            if res.shape[1] == o.shape[1]:
                o = o + res
        return o


class DurationPredictor(nn.Module):
    """
    Per-token scalar predictor (log-duration or pitch).

    Conv-ReLU-LayerNorm stack, port of coqui's FastPitch duration/pitch
    predictor head.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 256,
                 kernel_size: int = 3, dropout: float = 0.1, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden_channels
            self.layers.append(
                nn.Conv1d(in_c, hidden_channels, kernel_size, padding=kernel_size // 2)
            )
            self.norms.append(nn.LayerNorm(hidden_channels))
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Conv1d(hidden_channels, 1, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T], mask: [B, 1, T] -> [B, 1, T]."""
        o = x
        for conv, norm in zip(self.layers, self.norms):
            o = F.relu(conv(o * mask))
            o = norm(o.transpose(1, 2)).transpose(1, 2)
            o = self.dropout(o)
        return self.proj(o * mask) * mask
