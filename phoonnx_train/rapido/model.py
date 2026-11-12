"""Rápido acoustic network: token ids in, 64-dim VoxCPM2 AudioVAE latents at 25 Hz out.

Three stages, each usable on its own:

* text stage: a ConvNeXt + attention text encoder and a per-token duration predictor
  (``RapidoNet.text_stage``);
* aligner: a windowed Gaussian cross-attention that turns token features and the word
  and frame layout into a per-frame condition (``RapidoModel.condition``);
* velocity backbone: a DiT with shared adaptive LayerNorm that predicts the
  flow-matching velocity over normalized latents (``RapidoModel.velocity``).

Parameter names match the released ``rapido.pt`` checkpoint (``ema`` state dict), so
``load_state_dict(strict=True)`` loads it. All stages take padded batches with masks.
"""
import math
from dataclasses import dataclass, fields
from typing import Any, Collection, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phoonnx.engines.rapido import frame_layout, ode_times, word_frame_counts, word_layout

REPO_ID = "edwixx/rapido-tts"
CHECKPOINT_FILE = "rapido.pt"


@dataclass(frozen=True)
class RapidoConfig:
    """Network hyper-parameters; the defaults are those of the released checkpoint."""

    d: int = 384
    n_heads: int = 6
    n_layers: int = 8
    text_conv: int = 4
    text_attn: int = 1
    ff_mult: int = 4
    align_heads: int = 4
    dur_hidden: int = 256
    pos_scale: float = 24.0
    lookback: int = 1
    lookahead: int = 1
    letter_pos: bool = False
    latent_dim: int = 64
    shared_ada: bool = True

    def __post_init__(self):
        if self.letter_pos:
            raise ValueError("letter_pos=True aligner positions are not supported")

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "RapidoConfig":
        """Read a checkpoint ``cfg`` dict; keys that are not network shape are ignored."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in cfg.items() if k in names})


def rope(pos: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotary cos/sin tables for (possibly fractional) positions ``pos``."""
    inv = 1.0 / (10000.0 ** (torch.arange(dim // 2, device=pos.device).float() / (dim // 2)))
    ang = pos[..., None].float() * inv
    ang = torch.cat([ang, ang], -1)
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat([-b, a], -1) * sin


class SwiGLU(nn.Module):
    def __init__(self, d: int, mult: int = 4):
        super().__init__()
        hidden = int(d * mult * 2 / 3)
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = (y.view(b, t, self.heads, self.head_dim).transpose(1, 2)
                   for y in self.qkv(x).chunk(3, -1))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        return self.proj(x.transpose(1, 2).reshape(b, t, d))


class ConvNeXtBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.dw = nn.Conv1d(d, d, 7, padding=3, groups=d)
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, 2 * d)
        self.pw2 = nn.Linear(2 * d, d)
        self.gamma = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # padded positions read as zeros in the depthwise convolution
        y = self.norm(self.dw((x * mask[..., None]).transpose(1, 2)).transpose(1, 2))
        return x + self.gamma * self.pw2(F.gelu(self.pw1(y)))


class TextBlock(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.ff = Attention(d, heads), SwiGLU(d)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x), cos, sin, mask)
        return x + self.ff(self.n2(x))


class TextEncoder(nn.Module):
    def __init__(self, n_vocab: int, d: int, n_conv: int, n_attn: int, heads: int):
        super().__init__()
        self.head_dim = d // heads
        self.emb = nn.Embedding(n_vocab, d, padding_idx=0)
        self.conv = nn.ModuleList(ConvNeXtBlock(d) for _ in range(n_conv))
        self.attn = nn.ModuleList(TextBlock(d, heads) for _ in range(n_attn))
        self.norm = nn.LayerNorm(d)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        for block in self.conv:
            x = block(x, mask)
        cos, sin = rope(torch.arange(ids.shape[1], device=ids.device), self.head_dim)
        for block in self.attn:
            x = block(x, cos, sin, mask)
        return self.norm(x)


def masked_group_norm(x: torch.Tensor, norm: nn.GroupNorm, m: torch.Tensor) -> torch.Tensor:
    """GroupNorm(1, C) with statistics over the real positions only."""
    count = m.sum((1, 2), keepdim=True) * x.shape[1]
    mean = (x * m).sum((1, 2), keepdim=True) / count
    var = (((x - mean) * m) ** 2).sum((1, 2), keepdim=True) / count
    return (x - mean) / torch.sqrt(var + norm.eps) * norm.weight[None, :, None] + norm.bias[None, :, None]


class DurationPredictor(nn.Module):
    """Frames per token. ``net`` keeps the checkpoint's Sequential indices (0, 2, 4, 6)."""

    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d, hidden, 3, padding=1), nn.SiLU(), nn.GroupNorm(1, hidden), nn.Identity(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.SiLU(), nn.GroupNorm(1, hidden))
        self.out = nn.Linear(hidden, 1)

    def log_duration(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """log(1 + frames) per token before any clamp, 0 at padding; the training target."""
        m = mask[:, None].float()
        x = masked_group_norm(F.silu(self.net[0](h.transpose(1, 2) * m)), self.net[2], m) * m
        x = masked_group_norm(F.silu(self.net[4](x)), self.net[6], m)
        return self.out(x.transpose(1, 2)).squeeze(-1).masked_fill(~mask, 0.0)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        log_d = self.log_duration(h, mask)
        return torch.expm1(log_d.clamp(max=6.0)).clamp(min=1e-3) * mask.float()


class Aligner(nn.Module):
    """Frame queries attend to the tokens of their own word and its neighbours, with a
    Gaussian bias on the distance between frame and token positions."""

    def __init__(self, d: int, heads: int, pos_scale: float, lookback: int, lookahead: int):
        super().__init__()
        self.heads, self.head_dim = heads, d // heads
        self.pos_scale, self.lookback, self.lookahead = pos_scale, lookback, lookahead
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.frame_q = nn.Parameter(torch.zeros(1, 1, d))
        self.log_sigma = nn.Parameter(torch.zeros(heads))
        self.bias_w = nn.Parameter(torch.zeros(heads))
        self.log_temp = nn.Parameter(torch.zeros(()))

    def forward(self, h: torch.Tensor, word: torch.Tensor, token_pos: torch.Tensor,
                frame_word: torch.Tensor, frame_pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, length, d = h.shape
        frames = frame_word.shape[1]
        token_at = word.clamp(min=0).float() + token_pos
        frame_at = frame_word.float() + frame_pos
        q = self.q(self.frame_q.expand(b, frames, d)).view(b, frames, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(h).view(b, length, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(h).view(b, length, self.heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, *(t[:, None] for t in rope(frame_at * self.pos_scale, self.head_dim)))
        k = apply_rope(k, *(t[:, None] for t in rope(token_at * self.pos_scale, self.head_dim)))
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim) * self.log_temp.exp()
        sig2 = (self.log_sigma.exp() ** 2).view(1, self.heads, 1, 1)
        dist2 = ((frame_at[:, :, None] - token_at[:, None, :]) ** 2)[:, None]
        logits = logits - self.bias_w.view(1, self.heads, 1, 1) * dist2 / (2 * sig2)
        rel = word[:, None, :] - frame_word[:, :, None]
        allow = (rel >= -self.lookback) & (rel <= self.lookahead) & mask[:, None, :] & (word[:, None, :] >= 0)
        attn = logits.masked_fill(~allow[:, None], -1e4).softmax(-1) * allow[:, None]
        return self.o((attn @ v).transpose(1, 2).reshape(b, frames, d))


class DiTBlock(nn.Module):
    def __init__(self, d: int, heads: int, ff_mult: int, shared: bool):
        super().__init__()
        self.shared = shared
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn, self.ff = Attention(d, heads), SwiGLU(d, ff_mult)
        if shared:
            self.ada_offset = nn.Parameter(torch.zeros(6, d))
        else:
            self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))

    def forward(self, x: torch.Tensor, c: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        p = (c + self.ada_offset[None]).unbind(1) if self.shared else self.ada(c).chunk(6, -1)
        sa, ga, aa, sf, gf, af = (t.unsqueeze(1) for t in p)
        x = x + aa * self.attn(self.n1(x) * (1 + ga) + sa, cos, sin, mask)
        return x + af * self.ff(self.n2(x) * (1 + gf) + sf)


class TimestepEmbed(nn.Module):
    def __init__(self, d: int, freq: int = 256):
        super().__init__()
        self.freq = freq
        self.mlp = nn.Sequential(nn.Linear(freq, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq // 2
        a = t[:, None] * 1000.0 * torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)[None]
        return self.mlp(torch.cat([a.cos(), a.sin()], -1))


class RapidoNet(nn.Module):
    """Text encoder, duration predictor, aligner and DiT velocity backbone."""

    def __init__(self, config: RapidoConfig, n_vocab: int):
        super().__init__()
        d = config.d
        self.d, self.head_dim, self.shared = d, d // config.n_heads, config.shared_ada
        self.text = TextEncoder(n_vocab, d, config.text_conv, config.text_attn, config.n_heads)
        self.chardur = DurationPredictor(d, config.dur_hidden)
        self.aligner = Aligner(d, config.align_heads, config.pos_scale, config.lookback, config.lookahead)
        self.in_proj = nn.Linear(config.latent_dim, d)
        self.t_embed = TimestepEmbed(d)
        if config.shared_ada:
            self.ada_shared = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        self.blocks = nn.ModuleList(DiTBlock(d, config.n_heads, config.ff_mult, config.shared_ada)
                                    for _ in range(config.n_layers))
        self.norm_out = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out_proj = nn.Linear(d, config.latent_dim)

    def text_stage(self, ids: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Token ids -> token features [B,L,d] and frames per token [B,L]."""
        h = self.text(ids, mask)
        return h, self.chardur(h, mask)

    def backbone(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
        """Velocity [B,T,latent] of normalized latents ``x`` at flow time ``t`` [B]."""
        cos, sin = rope(torch.arange(x.shape[1], device=x.device), self.head_dim)
        c = self.t_embed(t)
        bc = self.ada_shared(c).view(-1, 6, self.d) if self.shared else c
        x = self.in_proj(x) + cond
        for block in self.blocks:
            x = block(x, bc, cos, sin, mask)
        s, g = self.ada_out(c).chunk(2, -1)
        return self.out_proj(self.norm_out(x) * (1 + g.unsqueeze(1)) + s.unsqueeze(1))


def token_positions(dur: torch.Tensor, mask: torch.Tensor, word: torch.Tensor,
                    word_start: torch.Tensor) -> torch.Tensor:
    """Each token's centre as a fraction of its word's duration, in [0, 1]."""
    c = dur.clamp(min=1e-4) * mask
    done = c.cumsum(-1)
    w = word.clamp(min=0)
    total = torch.zeros_like(c).scatter_add_(1, w, c).gather(1, w)
    return ((done - (done - c).gather(1, word_start) - 0.5 * c) / total.clamp(min=1e-8)).clamp(0.0, 1.0) * mask


class RapidoModel(nn.Module):
    """The acoustic network plus speaker embeddings and the latent statistics."""

    def __init__(self, config: RapidoConfig, n_vocab: int, n_speakers: int):
        super().__init__()
        self.config = config
        self.net = RapidoNet(config, n_vocab)
        self.spk = nn.Embedding(n_speakers, config.d)
        self.register_buffer("lat_mean", torch.zeros(config.latent_dim))
        self.register_buffer("lat_std", torch.ones(config.latent_dim))

    @classmethod
    def from_checkpoint(cls, checkpoint: Dict[str, Any]) -> "RapidoModel":
        """Build from a released-format checkpoint dict (``cfg``, ``vocab``, ``spk_order``,
        ``ema``) and load its weights strictly."""
        model = cls(RapidoConfig.from_dict(checkpoint["cfg"]), len(checkpoint["vocab"]),
                    len(checkpoint["spk_order"]))
        model.load_state_dict({k: v.float() for k, v in checkpoint["ema"].items()}, strict=True)
        return model.eval()

    def condition(self, h: torch.Tensor, dur: torch.Tensor, mask: torch.Tensor, word: torch.Tensor,
                  word_start: torch.Tensor, frame_word: torch.Tensor, frame_pos: torch.Tensor,
                  speaker: torch.Tensor) -> torch.Tensor:
        """Per-frame condition [B,T,d]: aligned token features plus the speaker embedding.

        ``word`` is -1 at padded tokens; ``dur`` may be predicted or forced durations.
        """
        token_pos = token_positions(dur, mask, word, word_start)
        aligned = self.net.aligner(h, word, token_pos, frame_word, frame_pos, mask)
        return aligned + self.spk(speaker)[:, None]

    def velocity(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor,
                 frame_mask: torch.Tensor) -> torch.Tensor:
        """Flow-matching velocity over normalized latents."""
        return self.net.backbone(x, cond, t, frame_mask)

    @torch.inference_mode()
    def generate(self, ids: Sequence[int], boundary_ids: Collection[int], speaker: int = 0,
                 steps: int = 8, guidance: float = 4.0, length_scale: float = 1.0, seed: int = 0,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Token ids -> denormalized latents [T, latent_dim], by the midpoint ODE with
        classifier-free guidance. The default noise is the ONNX adapter's: numpy
        ``default_rng(seed)``; pass ``noise`` [1,T,latent_dim] to use other noise."""
        device = self.lat_mean.device
        text_ids = torch.as_tensor(np.asarray(ids, dtype=np.int64), device=device)[None]
        mask = torch.ones_like(text_ids, dtype=torch.bool)
        h, dur = self.net.text_stage(text_ids, mask)
        word, word_start = word_layout(ids, boundary_ids)
        counts = word_frame_counts(dur[0].cpu().numpy() * np.float32(length_scale), word)
        frame_word, frame_pos = frame_layout(counts)
        as_tensor = lambda a: torch.from_numpy(a)[None].to(device)
        cond = self.condition(h, dur, mask, as_tensor(word), as_tensor(word_start), as_tensor(frame_word),
                              as_tensor(frame_pos), torch.tensor([speaker], device=device))
        frame_mask = torch.ones(1, len(frame_word), dtype=torch.bool, device=device)
        if noise is None:
            rng = np.random.default_rng(seed)
            noise = torch.from_numpy(rng.standard_normal((1, len(frame_word), self.config.latent_dim))
                                     .astype(np.float32))
        x = noise.to(device)

        def v(x: torch.Tensor, t: float) -> torch.Tensor:
            tt = torch.full((1,), t, device=device)
            vc = self.velocity(x, cond, tt, frame_mask)
            vu = self.velocity(x, torch.zeros_like(cond), tt, frame_mask)
            return vu + guidance * (vc - vu)

        times = ode_times(steps).tolist()
        for a, b in zip(times[:-1], times[1:]):
            x = x + (b - a) * v(x + (b - a) / 2 * v(x, a), (a + b) / 2)
        return x[0] * self.lat_std + self.lat_mean


def load_checkpoint(path: Optional[str] = None) -> Dict[str, Any]:
    """Read a released-format Rápido checkpoint; the Hub's ``rapido.pt`` when no path is given."""
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(REPO_ID, CHECKPOINT_FILE)
    return torch.load(path, map_location="cpu", weights_only=False)
