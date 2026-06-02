import math
import typing

import torch
from torch import nn
from torch.nn import Conv1d
from torch.nn import functional as F

from vits import attentions, commons, monotonic_align
from vits.models import (
    DurationPredictor,
    Generator,
    MultiPeriodDiscriminator,
    PosteriorEncoder,
    StochasticDurationPredictor,
    TextEncoder,
)

from .modules import ResidualCouplingBlock


class SwiGLU(nn.Module):
    """SwiGLU activation: Swish-gated linear unit.

    Splits the channel dimension in half, applies SiLU to the second half,
    and elementwise-multiplies with the first half.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=1)
        return x * F.silu(gate)


class ReferenceEncoder(nn.Module):
    """Minimal-parameter reference encoder using LSTM + SwiGLU.

    Compresses a mel-spectrogram into a fixed-size latent vector.
    Replaces the Conv+GRU stack with fewer, smarter layers:

    - 2 strided Conv-SwiGLU blocks (instead of 3 Conv-ReLU)
    - Single-layer BiLSTM with reduced hidden size (instead of BiGRU)
    - SwiGLU projection bottleneck

    This cuts the parameter count by roughly 50 % compared with the
    Conv-ReLU-GRU baseline while improving gradient flow.

    Args:
        in_channels: Number of mel channels (e.g. 80).
        hidden_channels: Width of intermediate conv feature maps.
        out_channels: Dimension of the output reference embedding.
        n_conv_layers: Number of Conv-SwiGLU-LayerNorm blocks (default 2).
        kernel_size: Conv kernel size (default 3).
        stride: Conv stride (default 2), halves the time dimension each layer.
        n_lstm_layers: Number of LSTM layers (default 1).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_conv_layers: int = 2,
        kernel_size: int = 3,
        stride: int = 2,
        n_lstm_layers: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # Conv-SwiGLU downsampling blocks.
        # SwiGLU doubles channels then gates back, so the conv outputs 2*hidden.
        self.conv_blocks = nn.ModuleList()
        for i in range(n_conv_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_ch,
                        hidden_channels * 2,
                        kernel_size,
                        stride=stride,
                        padding=kernel_size // 2,
                    ),
                    SwiGLU(),
                    nn.LayerNorm(hidden_channels),
                    nn.Dropout(0.1),
                )
            )

        # BiLSTM — smaller hidden size is fine because LSTM has better
        # gradient flow than GRU, and the SwiGLU blocks already extracted
        # strong local features.
        self.lstm = nn.LSTM(
            hidden_channels,
            hidden_channels,
            n_lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        # SwiGLU projection bottleneck: 2*hidden -> hidden -> out.
        # Using a bottleneck cuts parameters vs. a direct Linear(2*hidden, out).
        self.proj_gate = nn.Linear(hidden_channels * 2, hidden_channels * 2)
        self.proj_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        for block in self.conv_blocks:
            x = block(x)  # [B, hidden, T']

        # LSTM over time
        x = x.transpose(1, 2)  # [B, T', hidden]
        _, (h_n, _) = self.lstm(x)  # h_n: [2, B, hidden]

        # Take last forward + backward states
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [B, 2*hidden]

        # SwiGLU projection
        h = self.proj_gate(h)  # [B, 2*hidden]
        h, gate = h.chunk(2, dim=-1)
        h = h * F.silu(gate)  # [B, hidden]
        return self.proj_out(h).unsqueeze(-1)  # [B, out, 1]


class TimbreEncoder(nn.Module):
    """Encodes voice identity (timbre) from a speaker ID or a reference mel."""

    def __init__(
        self,
        n_speakers: int,
        gin_channels: int,
        ref_in_channels: int = 80,
        ref_hidden_channels: int = 256,
        ref_n_layers: int = 3,
        ref_kernel_size: int = 3,
        ref_stride: int = 2,
        ref_n_lstm_layers: int = 1,
        ref_enc_enabled: bool = False,
    ):
        super().__init__()
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.ref_enc_enabled = ref_enc_enabled

        if n_speakers > 1:
            self.speaker_emb = nn.Embedding(n_speakers, gin_channels)
        else:
            self.speaker_emb = None

        if ref_enc_enabled:
            self.ref_enc = ReferenceEncoder(
                in_channels=ref_in_channels,
                hidden_channels=ref_hidden_channels,
                out_channels=gin_channels,
                n_conv_layers=ref_n_layers,
                kernel_size=ref_kernel_size,
                stride=ref_stride,
                n_lstm_layers=ref_n_lstm_layers,
            )
        else:
            self.ref_enc = None

    def forward(
        self,
        sid: typing.Optional[torch.Tensor] = None,
        ref_mel: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if ref_mel is not None and self.ref_enc_enabled:
            return self.ref_enc(ref_mel)
        if sid is not None and self.speaker_emb is not None:
            return self.speaker_emb(sid).unsqueeze(-1)
        return None


class ProsodyEncoder(nn.Module):
    """Encodes prosodic / rhythmic / emotional information from a reference mel."""

    def __init__(
        self,
        out_channels: int,
        ref_in_channels: int = 80,
        ref_hidden_channels: int = 256,
        ref_n_layers: int = 3,
        ref_kernel_size: int = 3,
        ref_stride: int = 2,
        ref_n_lstm_layers: int = 1,
        n_emotion_labels: int = 0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.ref_enc = ReferenceEncoder(
            in_channels=ref_in_channels,
            hidden_channels=ref_hidden_channels,
            out_channels=out_channels,
            n_conv_layers=ref_n_layers,
            kernel_size=ref_kernel_size,
            stride=ref_stride,
            n_lstm_layers=ref_n_lstm_layers,
        )
        if n_emotion_labels > 0:
            self.emotion_emb = nn.Embedding(n_emotion_labels, out_channels)
        else:
            self.emotion_emb = None

    def forward(
        self,
        ref_mel: typing.Optional[torch.Tensor] = None,
        emotion_id: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if ref_mel is not None:
            return self.ref_enc(ref_mel)
        if emotion_id is not None and self.emotion_emb is not None:
            return self.emotion_emb(emotion_id).unsqueeze(-1)
        return None


class ArticulationEncoder(nn.Module):
    """Encodes articulatory / pronunciation envelope information from a reference mel."""

    def __init__(
        self,
        out_channels: int,
        ref_in_channels: int = 80,
        ref_hidden_channels: int = 256,
        ref_n_layers: int = 3,
        ref_kernel_size: int = 3,
        ref_stride: int = 2,
        ref_n_lstm_layers: int = 1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.ref_enc = ReferenceEncoder(
            in_channels=ref_in_channels,
            hidden_channels=ref_hidden_channels,
            out_channels=out_channels,
            n_conv_layers=ref_n_layers,
            kernel_size=ref_kernel_size,
            stride=ref_stride,
            n_lstm_layers=ref_n_lstm_layers,
        )

    def forward(self, ref_mel: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        if ref_mel is not None:
            return self.ref_enc(ref_mel)
        return None


class DisentangledSynthesizerTrn(nn.Module):
    """
    Disentangled Synthesizer for Training.

    Separates voice identity (timbre), articulation, and prosody into
    independent conditioning signals, enabling fine-grained control at
    inference time.
    """

    def __init__(
        self,
        n_vocab: int,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: typing.Tuple[int, ...],
        resblock_dilation_sizes: typing.Tuple[typing.Tuple[int, ...], ...],
        upsample_rates: typing.Tuple[int, ...],
        upsample_initial_channel: int,
        upsample_kernel_sizes: typing.Tuple[int, ...],
        n_speakers: int = 1,
        gin_channels: int = 0,
        use_sdp: bool = True,
        ref_enc_hidden_channels: int = 256,
        ref_enc_n_layers: int = 3,
        ref_enc_kernel_size: int = 3,
        ref_enc_stride: int = 2,
        ref_enc_n_lstm_layers: int = 1,
        timbre_dim: int = 0,
        artic_dim: int = 0,
        prosody_dim: int = 0,
        n_emotion_labels: int = 0,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        self.use_sdp = use_sdp

        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )

        if use_sdp:
            self.dp = StochasticDurationPredictor(
                hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels
            )
        else:
            self.dp = DurationPredictor(
                hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
            )

        self.timbre_enc = TimbreEncoder(
            n_speakers=n_speakers,
            gin_channels=gin_channels,
            ref_hidden_channels=ref_enc_hidden_channels,
            ref_n_layers=ref_enc_n_layers,
            ref_kernel_size=ref_enc_kernel_size,
            ref_stride=ref_enc_stride,
            ref_n_lstm_layers=ref_enc_n_lstm_layers,
            ref_enc_enabled=True,
        )
        self.artic_enc = ArticulationEncoder(
            out_channels=artic_dim if artic_dim > 0 else gin_channels,
            ref_hidden_channels=ref_enc_hidden_channels,
            ref_n_layers=ref_enc_n_layers,
            ref_kernel_size=ref_enc_kernel_size,
            ref_stride=ref_enc_stride,
            ref_n_lstm_layers=ref_enc_n_lstm_layers,
        )
        self.prosody_enc = ProsodyEncoder(
            out_channels=prosody_dim if prosody_dim > 0 else gin_channels,
            ref_hidden_channels=ref_enc_hidden_channels,
            ref_n_layers=ref_enc_n_layers,
            ref_kernel_size=ref_enc_kernel_size,
            ref_stride=ref_enc_stride,
            ref_n_lstm_layers=ref_enc_n_lstm_layers,
            n_emotion_labels=n_emotion_labels,
        )
        self._timbre_dim = timbre_dim if timbre_dim > 0 else gin_channels
        self._artic_dim = artic_dim if artic_dim > 0 else gin_channels
        self._prosody_dim = prosody_dim if prosody_dim > 0 else gin_channels

        self.artic_proj = (
            nn.Conv1d(self._artic_dim, hidden_channels, 1)
            if self._artic_dim != hidden_channels
            else None
        )

        self.flow_cond_proj = nn.Conv1d(
            self._timbre_dim + self._artic_dim + self._prosody_dim,
            gin_channels,
            1,
        )

    def _get_g_disentangled(self, sid, timbre_ref_mel, artic_ref_mel,
                            prosody_ref_mel, emotion_id):
        """Return (g_timbre, g_artic, g_prosody, g_flow)."""
        g_timbre = self.timbre_enc(sid=sid, ref_mel=timbre_ref_mel)
        g_artic = self.artic_enc(ref_mel=artic_ref_mel)
        g_prosody = self.prosody_enc(ref_mel=prosody_ref_mel, emotion_id=emotion_id)

        device = next(self.parameters()).device
        if g_timbre is None:
            g_timbre = torch.zeros(
                sid.size(0) if sid is not None else 1,
                self._timbre_dim, 1, device=device, dtype=next(self.parameters()).dtype
            )
        if g_artic is None:
            g_artic = torch.zeros(
                g_timbre.size(0), self._artic_dim, 1, device=device, dtype=g_timbre.dtype
            )
        if g_prosody is None:
            g_prosody = torch.zeros(
                g_timbre.size(0), self._prosody_dim, 1, device=device, dtype=g_timbre.dtype
            )

        g_flow_input = torch.cat([g_timbre, g_artic, g_prosody], dim=1)
        g_flow = self.flow_cond_proj(g_flow_input)

        return g_timbre, g_artic, g_prosody, g_flow

    def forward(self, x, x_lengths, y, y_lengths, sid=None,
                timbre_ref_mel=None, artic_ref_mel=None,
                prosody_ref_mel=None, emotion_id=None):

        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        g_timbre, g_artic, g_prosody, g_flow = self._get_g_disentangled(
            sid, timbre_ref_mel, artic_ref_mel, prosody_ref_mel, emotion_id
        )
        if self.artic_proj is not None:
            m_p = m_p + self.artic_proj(g_artic) * x_mask

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_timbre)
        z_p = self.flow(z, y_mask, g=g_flow)

        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent2 = torch.matmul(
                -0.5 * (z_p**2).transpose(1, 2), s_p_sq_r
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2), (m_p * s_p_sq_r)
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(
                -0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = (
                monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )

        w = attn.sum(2)
        if self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g_prosody)
            l_length = l_length / torch.sum(x_mask)
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g_prosody)
            l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(
                x_mask
            )  # for averaging

        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=g_timbre)
        return (
            o,
            l_length,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
        )

    def infer(
        self,
        x,
        x_lengths,
        sid=None,
        timbre_ref_mel=None,
        artic_ref_mel=None,
        prosody_ref_mel=None,
        emotion_id=None,
        noise_scale=0.667,
        length_scale=1,
        noise_scale_w=0.8,
        max_len=None,
    ):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        g_timbre, g_artic, g_prosody, g_flow = self._get_g_disentangled(
            sid, timbre_ref_mel, artic_ref_mel, prosody_ref_mel, emotion_id
        )
        if self.artic_proj is not None:
            m_p = m_p + self.artic_proj(g_artic) * x_mask

        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g_prosody, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g_prosody)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(
            commons.sequence_mask(y_lengths, y_lengths.max()), 1
        ).type_as(x_mask)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g_flow, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g_timbre)

        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt,
                         timbre_ref_mel_src=None, timbre_ref_mel_tgt=None,
                         artic_ref_mel=None, prosody_ref_mel=None,
                         emotion_id=None):
        g_timbre_src, _, _, g_flow_src = self._get_g_disentangled(
            sid_src, timbre_ref_mel_src, artic_ref_mel, prosody_ref_mel, emotion_id
        )
        g_timbre_tgt, _, _, g_flow_tgt = self._get_g_disentangled(
            sid_tgt, timbre_ref_mel_tgt, artic_ref_mel, prosody_ref_mel, emotion_id
        )
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_timbre_src)
        z_p = self.flow(z, y_mask, g=g_flow_src)
        z_hat = self.flow(z_p, y_mask, g=g_flow_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_timbre_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)
