"""
GlowTTS top-level generator.

Combines the text encoder (:mod:`phoonnx_train.glowtts.encoder`), the
invertible flow decoder (:mod:`phoonnx_train.glowtts.decoder`), and
Monotonic Alignment Search (:mod:`phoonnx_train.glowtts.monotonic_align`)
into the full GlowTTS generative model, reconstructed from the GlowTTS paper
(Kim et al. 2020).

Training (``forward``): the decoder flow maps the *ground-truth* mel
spectrogram to latent ``z`` (exact log-likelihood via change of variables).
MAS finds the best monotonic alignment between text-token priors and mel
frames under the current model, which both (a) supervises the duration
predictor (MSE on log-durations) and (b) expands the per-token prior
statistics into per-frame targets for the MLE loss.

Inference/export (``infer``): sample from the per-token Gaussian prior
(scaled by ``noise_scale``), expand it to mel-frame length using
predicted durations (scaled by ``length_scale``), then run the decoder flow
in reverse to produce a mel spectrogram.  This is the exact computation
``export_onnx`` traces for the ONNX graph consumed by
``phoonnx.engines.glowtts.GlowTTSAdapter``.
"""
import math
from typing import Optional, Tuple

import torch
from torch import nn

from phoonnx_train.vits import commons
from phoonnx_train.glowtts.encoder import TextEncoder
from phoonnx_train.glowtts.decoder import FlowDecoder
from phoonnx_train.glowtts.monotonic_align import maximum_path


class GlowTTSGenerator(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        n_mels: int = 80,
        n_speakers: int = 1,
        gin_channels: int = 0,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        filter_channels_dp: int = 256,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        prenet_n_layers: int = 3,
        # decoder / flow
        dec_hidden_channels: int = 192,
        dec_kernel_size: int = 5,
        dec_dilation_rate: int = 1,
        dec_n_blocks: int = 12,
        dec_n_layers: int = 4,
        n_sqz: int = 2,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_mels = n_mels
        self.n_speakers = n_speakers
        self.n_sqz = n_sqz

        eff_gin = gin_channels if n_speakers > 1 else 0
        self.gin_channels = eff_gin
        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, eff_gin)
            nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)

        self.encoder = TextEncoder(
            n_vocab, n_mels, hidden_channels, filter_channels, filter_channels_dp,
            n_heads, n_layers, kernel_size, p_dropout, prenet_n_layers=prenet_n_layers,
            gin_channels=eff_gin,
        )
        self.decoder = FlowDecoder(
            n_mels, dec_hidden_channels, dec_kernel_size, dec_dilation_rate,
            dec_n_blocks, dec_n_layers, n_sqz=n_sqz, gin_channels=eff_gin,
        )

    def _speaker_embed(self, sid: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.n_speakers <= 1 or sid is None:
            return None
        return self.emb_g(sid).unsqueeze(-1)  # [b, gin, 1]

    # ------------------------------------------------------------------
    # Training forward: exact NLL under the flow + MAS-derived duration loss
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        sid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        g = self._speaker_embed(sid)

        x_enc, m_p, logs_p, logw, x_mask = self.encoder(x, x_lengths, g=g)

        # pad mel length to a multiple of n_sqz so squeeze/unsqueeze round-trips
        y_max_length = (y.size(2) // self.n_sqz) * self.n_sqz
        y = y[:, :, :y_max_length]
        y_lengths = torch.clamp_max(y_lengths, y_max_length)
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1).type_as(x_enc)

        z, logdet = self.decoder(y, y_mask, g=g, reverse=False)

        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t_text]
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
            neg_cent2 = torch.matmul(-0.5 * (z**2).transpose(1, 2), s_p_sq_r)
            neg_cent3 = torch.matmul(z.transpose(1, 2), (m_p * s_p_sq_r))
            neg_cent4 = torch.sum(-0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4  # [b, t_mel, t_text]

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()  # [b,1,t_mel,t_text]

        # expand text-token prior stats to mel-frame targets via the hard alignment
        attn_t = attn.squeeze(1)  # [b, t_mel, t_text]
        m_p_frame = torch.matmul(attn_t, m_p.transpose(1, 2)).transpose(1, 2)  # [b, d, t_mel]
        logs_p_frame = torch.matmul(attn_t, logs_p.transpose(1, 2)).transpose(1, 2)

        logw_ = torch.log(1e-8 + torch.sum(attn_t, dim=1)).unsqueeze(1) * x_mask  # target log-duration

        return z, logdet, m_p_frame, logs_p_frame, logw, logw_, x_mask, y_mask

    # ------------------------------------------------------------------
    # Inference / ONNX export forward
    # ------------------------------------------------------------------

    def infer(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        sid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self._speaker_embed(sid)
        x_enc, m_p, logs_p, logw, x_mask = self.encoder(x, x_lengths, g=g)

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        # round each length up to a multiple of n_sqz so the squeeze/unsqueeze
        # round-trip in the flow decoder is exact and the returned lengths
        # match the mel frame axis. Kept as tensor ops (no .item()) so the
        # ONNX trace stays dynamic in output length.
        y_lengths = ((y_lengths + self.n_sqz - 1) // self.n_sqz) * self.n_sqz
        y_max_length = torch.max(y_lengths)

        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1).type_as(x_enc)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)  # [b,1,t_mel,t_text]
        attn = commons.generate_path(w_ceil, attn_mask)  # [b,1,t_mel,t_text]

        attn_t = attn.squeeze(1)  # [b, t_mel, t_text]
        m_p_frame = torch.matmul(attn_t, m_p.transpose(1, 2)).transpose(1, 2)
        logs_p_frame = torch.matmul(attn_t, logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p_frame + torch.exp(logs_p_frame) * torch.randn_like(m_p_frame) * noise_scale
        mel, _ = self.decoder(z_p, y_mask, g=g, reverse=True)
        return mel, y_lengths
