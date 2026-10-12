"""Text-to-latent module (conditional flow matching).

Flow matching learns a velocity field that transports Gaussian noise to the data
distribution along a straight path. Here the "data" is the compressed speech
latent, and the field is conditioned on a text embedding and a reference voice
style. At inference the field is integrated with a few Euler steps to turn noise
into a latent; at training we regress the known straight-line target velocity.

The style tokens (``style_ttl``) and text embedding are computed once and reused
across ``batch_expand`` independent noise/timestep draws per utterance.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from phoonnx_train.supertonic.config import TextToLatentConfig
from phoonnx_train.supertonic.layers import (
    ChannelNorm,
    ConvNeXtStack,
    FixedContextCrossAttention,
    RelPositionEncoder,
    RotaryAttention,
    StylePool,
    TimestepEmbedding,
)


class StyleEncoder(nn.Module):
    """Reference compressed latent -> ``n_style`` style tokens ``(B, n_style, style_dim)``."""

    def __init__(self, cfg: TextToLatentConfig):
        super().__init__()
        self.proj_in = nn.Linear(cfg.compressed_dim, cfg.style_dim)
        self.convnext = ConvNeXtStack(cfg.style_dim, cfg.style_convnext_ffn,
                                      cfg.style_convnext_kernel, (1,) * cfg.style_convnext_layers)
        self.pool = StylePool(cfg.style_dim, cfg.n_style, cfg.style_dim, cfg.style_heads)

    def forward(self, ref: torch.Tensor, ref_mask=None) -> torch.Tensor:
        x = self.proj_in(ref.transpose(1, 2)).transpose(1, 2)
        x = self.convnext(x, ref_mask)
        return self.pool(x, ref_mask)


class TextEncoder(nn.Module):
    """Char ids -> style-conditioned text embedding ``(B, char_dim, T)``."""

    def __init__(self, cfg: TextToLatentConfig, vocab_size: int):
        super().__init__()
        dim = cfg.char_dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.convnext = ConvNeXtStack(dim, cfg.convnext_ffn, cfg.convnext_kernel, cfg.vf_dilations)
        self.self_attn = RelPositionEncoder(dim, cfg.self_attn_ffn, cfg.self_attn_heads,
                                            cfg.self_attn_layers, cfg.self_attn_window)
        self.style_attn1 = FixedContextCrossAttention(dim, cfg.style_dim, cfg.prompt_heads, hidden=dim)
        self.style_attn2 = FixedContextCrossAttention(dim, cfg.style_dim, cfg.prompt_heads, hidden=dim)
        self.norm = ChannelNorm(dim)

    def forward(self, text_ids, text_mask, style_ttl) -> torch.Tensor:
        x = self.embed(text_ids).transpose(1, 2)
        conv = self.convnext(x, text_mask)
        x = (self.self_attn(conv, text_mask) + conv) * text_mask
        x = x + self.style_attn1(x, style_ttl)
        x = x + self.style_attn2(x, style_ttl)
        return self.norm(x) * text_mask


class UncondMasker(nn.Module):
    """Classifier-free-guidance dropout: replace text and/or style conditioning
    with learnable null tokens during training only."""

    def __init__(self, cfg: TextToLatentConfig):
        super().__init__()
        self.p_text = cfg.prob_text_uncond
        self.p_both = cfg.prob_both_uncond
        self.null_text = nn.Parameter(torch.randn(1, cfg.char_dim, 1) * 0.1)
        self.null_style = nn.Parameter(torch.randn(1, cfg.n_style, cfg.style_dim) * 0.1)

    def forward(self, text_emb, style_ttl):
        if not self.training:
            return text_emb, style_ttl
        b = text_emb.shape[0]
        r = torch.rand(b, device=text_emb.device)
        both = (r < self.p_both).view(b, 1, 1)
        text_only = ((r >= self.p_both) & (r < self.p_both + self.p_text)).view(b, 1, 1)
        text_emb = torch.where(both | text_only, self.null_text.expand(b, -1, text_emb.shape[-1]), text_emb)
        style_ttl = torch.where(both, self.null_style.expand(b, -1, -1), style_ttl)
        return text_emb, style_ttl


class VFBlock(nn.Module):
    """One repeated block of the velocity-field estimator: dilated ConvNeXt, time
    conditioning, text cross-attention (length-aware rotary), ConvNeXt, style
    cross-attention (tanh-bounded)."""

    def __init__(self, cfg: TextToLatentConfig):
        super().__init__()
        dim = cfg.vf_dim
        self.dilated = ConvNeXtStack(dim, cfg.vf_ffn, cfg.vf_kernel, cfg.vf_dilations)
        self.time_proj = nn.Linear(cfg.time_dim, dim)
        self.mid = ConvNeXtStack(dim, cfg.vf_ffn, cfg.vf_kernel, (1,))
        self.text_attn = RotaryAttention(dim, cfg.char_dim, cfg.vf_text_heads, units=dim, rotary_base=cfg.rotary_base)
        self.text_norm = ChannelNorm(dim)
        self.post = ConvNeXtStack(dim, cfg.vf_ffn, cfg.vf_kernel, (1,))
        self.style_attn = FixedContextCrossAttention(dim, cfg.style_dim, cfg.vf_style_heads, hidden=cfg.style_dim)
        self.style_norm = ChannelNorm(dim)
        self.gamma = cfg.rotary_scale

    def forward(self, x, t_emb, text_emb, style_ttl, latent_mask, text_mask):
        x = self.dilated(x, latent_mask)
        x = (x + self.time_proj(t_emb).unsqueeze(-1)) * latent_mask
        x = self.mid(x, latent_mask)
        q_len = latent_mask.sum(dim=(1, 2))
        k_len = text_mask.sum(dim=(1, 2))
        y = self.text_attn(x, text_emb, mask=text_mask, length_aware=True, gamma=self.gamma, q_len=q_len, k_len=k_len)
        x = self.text_norm(x + y) * latent_mask
        x = self.post(x, latent_mask)
        x = self.style_norm(x + self.style_attn(x, style_ttl))
        return x * latent_mask


class VFEstimator(nn.Module):
    def __init__(self, cfg: TextToLatentConfig):
        super().__init__()
        self.proj_in = nn.Linear(cfg.compressed_dim, cfg.vf_dim)
        self.time = TimestepEmbedding(cfg.time_dim, cfg.time_hidden)
        self.blocks = nn.ModuleList([VFBlock(cfg) for _ in range(cfg.vf_blocks)])
        self.final = ConvNeXtStack(cfg.vf_dim, cfg.vf_ffn, cfg.vf_kernel, (1,) * cfg.vf_final_layers)
        self.proj_out = nn.Linear(cfg.vf_dim, cfg.compressed_dim)

    def forward(self, noisy_latent, t, text_emb, style_ttl, latent_mask, text_mask):
        x = self.proj_in(noisy_latent.transpose(1, 2)).transpose(1, 2) * latent_mask
        t_emb = self.time(t)
        for block in self.blocks:
            x = block(x, t_emb, text_emb, style_ttl, latent_mask, text_mask)
        x = self.final(x, latent_mask)
        x = self.proj_out(x.transpose(1, 2)).transpose(1, 2)
        return x * latent_mask


class TextToLatentModel(nn.Module):
    def __init__(self, cfg: TextToLatentConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.style_encoder = StyleEncoder(cfg)
        self.text_encoder = TextEncoder(cfg, vocab_size)
        self.uncond = UncondMasker(cfg)
        self.vector_field = VFEstimator(cfg)

    def conditions(self, text_ids, text_mask, ref, ref_mask):
        style_ttl = self.style_encoder(ref, ref_mask)
        text_emb = self.text_encoder(text_ids, text_mask, style_ttl)
        return self.uncond(text_emb, style_ttl)

    def forward(self, noisy_latent, t, text_ids, text_mask, ref, ref_mask, latent_mask):
        text_emb, style_ttl = self.conditions(text_ids, text_mask, ref, ref_mask)
        return self.vector_field(noisy_latent, t, text_emb, style_ttl, latent_mask, text_mask)


def flow_matching_loss(model, z1, latent_mask, text_ids, text_mask, ref, ref_mask,
                       ref_time_mask, n_expand=1):
    """Optimal-transport conditional flow matching loss with context-sharing
    batch expansion. The reference-crop region is excluded from the loss."""
    cfg = model.cfg
    text_emb, style_ttl = model.conditions(text_ids, text_mask, ref, ref_mask)
    b, c, _ = z1.shape

    z1_e = z1.repeat_interleave(n_expand, dim=0)
    mask_e = latent_mask.repeat_interleave(n_expand, dim=0)
    rt_e = ref_time_mask.repeat_interleave(n_expand, dim=0)
    text_emb_e = text_emb.repeat_interleave(n_expand, dim=0)
    text_mask_e = text_mask.repeat_interleave(n_expand, dim=0)
    style_e = style_ttl.repeat_interleave(n_expand, dim=0)

    z0 = torch.randn_like(z1_e)
    t = torch.rand(b * n_expand, device=z1.device)
    tb = t.view(-1, 1, 1)
    sig = cfg.sigma_min
    zt = (1 - (1 - sig) * tb) * z0 + tb * z1_e
    target = z1_e - (1 - sig) * z0

    pred = model.vector_field(zt, t, text_emb_e, style_e, mask_e, text_mask_e)
    loss_mask = mask_e * (1 - rt_e)
    diff = (pred - target).abs() * loss_mask
    return diff.sum() / (loss_mask.sum().clamp_min(1.0) * c)
