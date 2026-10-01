"""Reusable neural building blocks for the SuperTonic stack.

Every module here works on ``(batch, channels, time)`` tensors and, where a
``mask`` is accepted, expects it shaped ``(batch, 1, time)`` with ``1`` marking
valid positions and ``0`` marking padding. Masking padded frames repeatedly
inside each block keeps padded batches numerically identical to single-utterance
batches, which matters as soon as real (ragged) data is used.

These are original implementations of standard components — Vocos-style ConvNeXt
blocks, rotary multi-head attention, the VITS/Glow-TTS windowed
relative-position self-attention, a fixed-context cross-attention pooler, and a
sinusoidal timestep embedding.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """``(B,)`` valid-lengths -> ``(B, 1, max_len)`` float mask."""
    if max_len is None:
        max_len = int(lengths.max().item())
    steps = torch.arange(max_len, device=lengths.device)
    valid = steps.unsqueeze(0) < lengths.unsqueeze(1)
    return valid.float().unsqueeze(1)


class ConvNeXtBlock(nn.Module):
    """Depthwise-conv ConvNeXt block (Vocos flavour) over ``(B, C, T)``.

    ``causal=True`` pads only on the left so no future frame leaks into the
    present one — used by the streaming waveform decoder. Padding uses replicate
    mode, matching the released graphs' edge-padding.
    """

    def __init__(self, dim: int, ffn_dim: int, kernel: int = 7, dilation: int = 1,
                 causal: bool = False, scale_init: float = 1e-6):
        super().__init__()
        span = (kernel - 1) * dilation
        self.left = span if causal else span // 2
        self.right = 0 if causal else span - span // 2
        self.depthwise = nn.Conv1d(dim, dim, kernel, groups=dim, dilation=dilation)
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, ffn_dim)
        self.down = nn.Linear(ffn_dim, dim)
        self.scale = nn.Parameter(scale_init * torch.ones(dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            x = x * mask
        skip = x
        x = F.pad(x, (self.left, self.right), mode="replicate")
        x = self.depthwise(x)
        if mask is not None:
            x = x * mask
        x = x.transpose(1, 2)
        x = self.down(F.gelu(self.up(self.norm(x))))
        x = (self.scale * x).transpose(1, 2)
        out = skip + x
        return out * mask if mask is not None else out


class ConvNeXtStack(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, kernel: int, dilations, causal: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock(dim, ffn_dim, kernel, d, causal=causal) for d in dilations]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask)
        return x


class ChannelNorm(nn.Module):
    """LayerNorm across the channel axis of a ``(B, C, T)`` tensor."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


class RotaryAttention(nn.Module):
    """Multi-head attention over ``(B, C, T)`` supporting cross-attention with
    length-normalised rotary positions on the query and key axes independently.

    ``length_aware`` divides each sample's positions by its own true (unpadded)
    length so text->latent cross-attention keeps a monotonic diagonal even when
    the two sequences differ in length.
    """

    def __init__(self, q_dim: int, kv_dim: int, n_heads: int, units: int | None = None,
                 rotary_base: float | None = None):
        super().__init__()
        units = units or q_dim
        assert units % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = units // n_heads
        self.q = nn.Linear(q_dim, units)
        self.k = nn.Linear(kv_dim, units)
        self.v = nn.Linear(kv_dim, units)
        self.o = nn.Linear(units, q_dim)
        if rotary_base:
            inv = 1.0 / (rotary_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            self.register_buffer("inv_freq", inv, persistent=False)
        else:
            self.inv_freq = None

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def _cos_sin(self, positions: torch.Tensor, dtype):
        freqs = positions.unsqueeze(-1) * self.inv_freq.to(positions.device)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, x_q, x_kv, mask=None, length_aware=False, gamma=1.0,
                q_len=None, k_len=None):
        q = self._heads(self.q(x_q.transpose(1, 2)))
        k = self._heads(self.k(x_kv.transpose(1, 2)))
        v = self._heads(self.v(x_kv.transpose(1, 2)))

        if self.inv_freq is not None:
            tq, tk = q.shape[2], k.shape[2]
            dev = q.device
            if length_aware:
                ql = (q_len if q_len is not None else torch.full((q.shape[0],), tq, device=dev)).clamp(min=1).float()
                kl = (k_len if k_len is not None else torch.full((k.shape[0],), tk, device=dev)).clamp(min=1).float()
                pos_q = gamma * torch.arange(tq, device=dev).float().unsqueeze(0) / ql.unsqueeze(1)
                pos_k = gamma * torch.arange(tk, device=dev).float().unsqueeze(0) / kl.unsqueeze(1)
                cos_q, sin_q = self._cos_sin(pos_q, q.dtype)
                cos_k, sin_k = self._cos_sin(pos_k, q.dtype)
                cos_q, sin_q, cos_k, sin_k = (t.unsqueeze(1) for t in (cos_q, sin_q, cos_k, sin_k))
            else:
                pos_q = torch.arange(tq, device=dev).float()
                pos_k = torch.arange(tk, device=dev).float()
                cos_q, sin_q = self._cos_sin(pos_q, q.dtype)
                cos_k, sin_k = self._cos_sin(pos_k, q.dtype)
                cos_q, sin_q, cos_k, sin_k = (t.unsqueeze(0).unsqueeze(0) for t in (cos_q, sin_q, cos_k, sin_k))
            q = q * cos_q + _rotate_half(q) * sin_q
            k = k * cos_k + _rotate_half(k) * sin_k

        bias = None
        if mask is not None:
            bias = torch.zeros_like(mask, dtype=q.dtype).masked_fill(mask < 0.5, float("-inf")).unsqueeze(1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = out.transpose(1, 2).reshape(x_q.shape[0], -1, self.n_heads * self.head_dim)
        return self.o(out).transpose(1, 2)


class FixedContextCrossAttention(nn.Module):
    """Cross-attention from a ``(B, C_x, T)`` stream onto a fixed-length context
    ``(B, T_ctx, C_ctx)`` (e.g. the style token bank). The context is re-projected
    into keys and values here. ``tanh_key`` bounds the key with tanh before the
    score (used for style conditioning). The score is scaled by ``sqrt(hidden)``.
    """

    def __init__(self, x_dim: int, ctx_dim: int, n_heads: int, hidden: int | None = None,
                 tanh_key: bool = True):
        super().__init__()
        hidden = hidden or ctx_dim
        assert hidden % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.hidden = hidden
        self.tanh_key = tanh_key
        self.q = nn.Linear(x_dim, hidden)
        self.k = nn.Linear(ctx_dim, hidden)
        self.v = nn.Linear(ctx_dim, hidden)
        self.o = nn.Linear(hidden, x_dim)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self._heads(self.q(x.transpose(1, 2)))
        k = self._heads(self.k(ctx))
        v = self._heads(self.v(ctx))
        key = torch.tanh(k) if self.tanh_key else k
        scores = torch.matmul(q, key.transpose(-2, -1)) / math.sqrt(self.hidden)
        if ctx_mask is not None:
            scores = scores.masked_fill(ctx_mask.view(x.shape[0], 1, 1, -1) < 0.5, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(x.shape[0], -1, self.n_heads * self.head_dim)
        return self.o(out).transpose(1, 2)


def _flatten_pad(shape):
    return [v for pair in reversed(shape) for v in pair]


class RelPositionSelfAttention(nn.Module):
    """Windowed relative-position multi-head self-attention (Shaw et al. 2018),
    as used in VITS/Glow-TTS. Learns a pair of relative-position embedding banks
    of width ``2*window+1`` and adds a relative-position score term to the
    standard dot-product attention.
    """

    def __init__(self, dim: int, n_heads: int, window: int = 4):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.dk = dim // n_heads
        self.window = window
        self.q = nn.Conv1d(dim, dim, 1)
        self.k = nn.Conv1d(dim, dim, 1)
        self.v = nn.Conv1d(dim, dim, 1)
        self.out = nn.Conv1d(dim, dim, 1)
        std = self.dk ** -0.5
        self.rel_k = nn.Parameter(torch.randn(1, 2 * window + 1, self.dk) * std)
        self.rel_v = nn.Parameter(torch.randn(1, 2 * window + 1, self.dk) * std)

    def _slice_rel(self, emb: torch.Tensor, length: int) -> torch.Tensor:
        pad = max(length - (self.window + 1), 0)
        start = max((self.window + 1) - length, 0)
        end = start + 2 * length - 1
        if pad > 0:
            emb = F.pad(emb, _flatten_pad([[0, 0], [pad, pad], [0, 0]]))
        return emb[:, start:end]

    def _rel_to_abs(self, x: torch.Tensor) -> torch.Tensor:
        b, h, length, _ = x.shape
        x = F.pad(x, _flatten_pad([[0, 0], [0, 0], [0, 0], [0, 1]]))
        flat = x.view(b, h, length * 2 * length)
        flat = F.pad(flat, _flatten_pad([[0, 0], [0, 0], [0, length - 1]]))
        return flat.view(b, h, length + 1, 2 * length - 1)[:, :, :length, length - 1:]

    def _abs_to_rel(self, x: torch.Tensor) -> torch.Tensor:
        b, h, length, _ = x.shape
        x = F.pad(x, _flatten_pad([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        flat = x.view(b, h, length ** 2 + length * (length - 1))
        flat = F.pad(flat, _flatten_pad([[0, 0], [0, 0], [length, 0]]))
        return flat.view(b, h, length, 2 * length)[:, :, :, 1:]

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, c, t = x.shape
        q = self.q(x).view(b, self.n_heads, self.dk, t).transpose(2, 3)
        k = self.k(x).view(b, self.n_heads, self.dk, t).transpose(2, 3)
        v = self.v(x).view(b, self.n_heads, self.dk, t).transpose(2, 3)

        qs = q / math.sqrt(self.dk)
        scores = torch.matmul(qs, k.transpose(-2, -1))
        rel_k = self._slice_rel(self.rel_k, t)
        rel_scores = torch.matmul(qs, rel_k.unsqueeze(0).transpose(-2, -1))
        scores = scores + self._rel_to_abs(rel_scores)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) < 0.5, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        rel_v = self._slice_rel(self.rel_v, t)
        out = torch.matmul(attn, v) + torch.matmul(self._abs_to_rel(attn), rel_v.unsqueeze(0))
        out = out.transpose(2, 3).reshape(b, c, t)
        return self.out(out)


class RelPositionEncoder(nn.Module):
    """Stack of relative-position self-attention + conv-FFN layers, each with a
    residual post-norm (VITS/Glow-TTS encoder layout)."""

    def __init__(self, dim: int, ffn_dim: int, n_heads: int, n_layers: int, window: int = 4):
        super().__init__()
        self.attn = nn.ModuleList([RelPositionSelfAttention(dim, n_heads, window) for _ in range(n_layers)])
        self.norm1 = nn.ModuleList([ChannelNorm(dim) for _ in range(n_layers)])
        self.ffn = nn.ModuleList(
            [nn.ModuleDict({"c1": nn.Conv1d(dim, ffn_dim, 1), "c2": nn.Conv1d(ffn_dim, dim, 1)})
             for _ in range(n_layers)]
        )
        self.norm2 = nn.ModuleList([ChannelNorm(dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for attn, n1, ffn, n2 in zip(self.attn, self.norm1, self.ffn, self.norm2):
            x = n1(x + attn(x, mask))
            h = x * mask if mask is not None else x
            h = F.relu(ffn["c1"](h))
            h = h * mask if mask is not None else h
            h = ffn["c2"](h)
            x = n2(x + h)
            if mask is not None:
                x = x * mask
        return x


class StylePool(nn.Module):
    """Pools a variable-length reference sequence into ``n_style`` fixed tokens
    with a learnable query bank, via one cross-attention pass."""

    def __init__(self, input_dim: int, n_style: int, dim: int, n_heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, n_style, dim) * 0.02)
        self.pool = FixedContextCrossAttention(dim, input_dim, n_heads, hidden=dim, tanh_key=False)

    def forward(self, ref: torch.Tensor, ref_mask: torch.Tensor | None = None) -> torch.Tensor:
        b = ref.shape[0]
        q = self.query.expand(b, -1, -1).transpose(1, 2)
        ctx_mask = ref_mask.squeeze(1) if ref_mask is not None else None
        out = self.pool(q, ref.transpose(1, 2), ctx_mask=ctx_mask)
        return out.transpose(1, 2)  # (B, n_style, dim)


class FlattenStylePool(nn.Module):
    """Pools a reference sequence into a single flat vector by concatenating
    ``n_style`` pooled tokens along the channel axis (duration predictor)."""

    def __init__(self, input_dim: int, n_style: int, dim: int, value_dim: int, n_heads: int):
        super().__init__()
        self.n_style = n_style
        self.query = nn.Parameter(torch.randn(1, n_style, dim) * 0.02)
        self.attn = RotaryAttention(dim, input_dim, n_heads, units=dim)
        self.proj = nn.Linear(dim, value_dim // n_style)

    def forward(self, ref: torch.Tensor, ref_mask: torch.Tensor | None = None) -> torch.Tensor:
        b = ref.shape[0]
        q = self.query.expand(b, -1, -1).transpose(1, 2)
        pooled = self.attn(q, ref, mask=ref_mask).transpose(1, 2)
        pooled = self.proj(pooled)
        return pooled.reshape(b, -1)


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep features expanded to ``hidden`` and projected back to
    ``dim`` (Grad-TTS style)."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)
