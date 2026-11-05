"""
Alignment network for unsupervised duration learning.

Adapted from coqui-ai/TTS ``TTS/tts/layers/generic/aligner.py``
(© Coqui GmbH, Mozilla Public License 2.0), itself following
"One TTS Alignment To Rule Them All" (https://arxiv.org/abs/2108.10447).
"""
from typing import Optional, Tuple

import torch
from torch import nn


class AlignmentNetwork(nn.Module):
    """
    Learns a soft text↔mel alignment; binarized with monotonic alignment
    search (see :func:`phoonnx_train.fastpitch.helpers.maximum_path`) to
    obtain per-token durations during training.
    """

    def __init__(self, in_query_channels: int = 80, in_key_channels: int = 256,
                 attn_channels: int = 80, temperature: float = 0.0005):
        super().__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)

        self.key_layer = nn.Sequential(
            nn.Conv1d(in_key_channels, in_key_channels * 2, kernel_size=3,
                      padding=1, bias=True),
            torch.nn.ReLU(),
            nn.Conv1d(in_key_channels * 2, attn_channels, kernel_size=1, bias=True),
        )
        self.query_layer = nn.Sequential(
            nn.Conv1d(in_query_channels, in_query_channels * 2, kernel_size=3,
                      padding=1, bias=True),
            torch.nn.ReLU(),
            nn.Conv1d(in_query_channels * 2, in_query_channels, kernel_size=1, bias=True),
            torch.nn.ReLU(),
            nn.Conv1d(in_query_channels, attn_channels, kernel_size=1, bias=True),
        )

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_prior: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        queries: mel frames [B, C_mel, T_de]
        keys:    encoder states [B, C_en, T_en]
        mask:    key mask [B, 1, T_en]

        Returns (attn [B, 1, T_de, T_en] softmax, attn_logp log-softmax).
        """
        key_out = self.key_layer(keys)
        query_out = self.query_layer(queries)
        attn_factor = (query_out[:, :, :, None] - key_out[:, :, None]) ** 2
        attn_logp = -self.temperature * attn_factor.sum(1, keepdim=True)
        if attn_prior is not None:
            attn_logp = self.log_softmax(attn_logp) + torch.log(attn_prior[:, None] + 1e-8)
        if mask is not None:
            attn_logp.data.masked_fill_(~mask.bool().unsqueeze(2), -float("inf"))
        attn = self.softmax(attn_logp)
        return attn, attn_logp
