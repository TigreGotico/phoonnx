"""Utterance-level duration predictor: estimates the total duration of the
synthesized speech directly from the text plus a reference voice style, avoiding
any phoneme-level alignment. A learnable "sentence token" prepended to the
character sequence gathers a fixed-size text embedding; a flattened reference
embedding is concatenated, and a small MLP predicts log-duration (exponentiated
to seconds).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from phoonnx_train.supertonic.config import DurationPredictorConfig
from phoonnx_train.supertonic.layers import ConvNeXtStack, FlattenStylePool, RelPositionEncoder


class SentenceEncoder(nn.Module):
    def __init__(self, cfg: DurationPredictorConfig, vocab_size: int):
        super().__init__()
        dim = cfg.char_dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.sentence_token = nn.Parameter(torch.randn(1, dim, 1) * 0.02)
        self.convnext = ConvNeXtStack(dim, cfg.convnext_ffn, cfg.convnext_kernel, (1,) * cfg.convnext_layers)
        self.self_attn = RelPositionEncoder(dim, cfg.self_attn_ffn, cfg.self_attn_heads,
                                            cfg.self_attn_layers, cfg.self_attn_window)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, text_ids, text_mask) -> torch.Tensor:
        b = text_ids.shape[0]
        x = self.embed(text_ids).transpose(1, 2)
        token = self.sentence_token.expand(b, -1, -1)
        x = torch.cat([token, x], dim=2)
        mask = torch.cat([torch.ones(b, 1, 1, device=text_mask.device), text_mask], dim=2)
        conv = self.convnext(x, mask)
        attn = self.self_attn(conv, mask)
        return self.proj((attn + conv)[:, :, 0])


class ReferenceEncoder(nn.Module):
    def __init__(self, cfg: DurationPredictorConfig):
        super().__init__()
        self.proj_in = nn.Linear(cfg.compressed_dim, cfg.style_dim)
        self.convnext = ConvNeXtStack(cfg.style_dim, cfg.style_convnext_ffn, 5, (1,) * cfg.style_convnext_layers)
        self.pool = FlattenStylePool(cfg.style_dim, cfg.n_style, cfg.style_dim,
                                     cfg.n_style * cfg.style_value_dim, cfg.style_heads)

    def forward(self, ref, ref_mask=None) -> torch.Tensor:
        x = self.proj_in(ref.transpose(1, 2)).transpose(1, 2)
        x = self.convnext(x, ref_mask)
        return self.pool(x, ref_mask)  # (B, n_style * style_value_dim)


class DurationPredictor(nn.Module):
    def __init__(self, cfg: DurationPredictorConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.text_encoder = SentenceEncoder(cfg, vocab_size)
        self.style_encoder = ReferenceEncoder(cfg)
        in_dim = cfg.char_dim + cfg.n_style * cfg.style_value_dim
        self.estimator = nn.Sequential(
            nn.Linear(in_dim, cfg.predictor_hidden), nn.PReLU(), nn.Linear(cfg.predictor_hidden, 1)
        )

    def log_duration_from_embeddings(self, text_emb, style_flat) -> torch.Tensor:
        return self.estimator(torch.cat([text_emb, style_flat], dim=-1)).squeeze(-1)

    def forward(self, text_ids, text_mask, ref, ref_mask=None) -> torch.Tensor:
        text_emb = self.text_encoder(text_ids, text_mask)
        style_flat = self.style_encoder(ref, ref_mask)
        return self.log_duration_from_embeddings(text_emb, style_flat).exp()


def duration_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()
