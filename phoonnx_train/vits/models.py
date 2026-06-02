import math
import typing

import torch
from torch import nn
from torch.nn import Conv1d, Conv2d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from . import attentions, commons, modules, monotonic_align
from .commons import get_padding, init_weights


class StochasticDurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2)).type_as(x) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum(
                (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
            )
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2])
                - logdet_tot_q
            )

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = (
                torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2])
                - logdet_tot
            )
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = torch.randn(x.size(0), 2, x.size(2)).type_as(x) * noise_scale

            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class DurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(
            commons.sequence_mask(x_lengths, x.size(2)), 1
        ).type_as(x)

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
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
                modules.ResidualCouplingLayer(
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
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, g_factors=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, g_factors=g_factors, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, g_factors=g_factors, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(
            commons.sequence_mask(x_lengths, x.size(2)), 1
        ).type_as(x)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel: int,
        resblock: typing.Optional[str],
        resblock_kernel_sizes: typing.Tuple[int, ...],
        resblock_dilation_sizes: typing.Tuple[typing.Tuple[int, ...], ...],
        upsample_rates: typing.Tuple[int, ...],
        upsample_initial_channel: int,
        upsample_kernel_sizes: typing.Tuple[int, ...],
        gin_channels: int = 0,
    ):
        super(Generator, self).__init__()
        self.LRELU_SLOPE = 0.1
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock_module = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock_module(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            x = up(x)
            xs = torch.zeros(1)
            for j, resblock in enumerate(self.resblocks):
                index = j - (i * self.num_kernels)
                if index == 0:
                    xs = resblock(x)
                elif (index > 0) and (index < self.num_kernels):
                    xs += resblock(x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ):
        super(DiscriminatorP, self).__init__()
        self.LRELU_SLOPE = 0.1
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        self.LRELU_SLOPE = 0.1
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ReferenceEncoder(nn.Module):
    """Compresses a mel-spectrogram reference into a fixed-size latent vector.

    Architecture: Conv-ReLU-BN stack followed by a bidirectional GRU, then a linear
    projection to the target dimension. Designed to be ONNX-friendly (no complex ops).

    Args:
        in_channels: Number of mel channels (e.g., 80).
        hidden_channels: Width of intermediate conv feature maps.
        out_channels: Dimension of the output reference embedding.
        n_conv_layers: Number of Conv-ReLU-LayerNorm blocks (default 3).
        kernel_size: Conv kernel size (default 3).
        stride: Conv stride (default 2), halves the time dimension each layer.
        n_gru_layers: Number of GRU layers (default 1).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_conv_layers: int = 3,
        kernel_size: int = 3,
        stride: int = 2,
        n_gru_layers: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_conv_layers = n_conv_layers

        # Build Conv stack: each layer halves time via stride=2
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_conv_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            padding = kernel_size // 2
            conv = nn.Conv1d(
                in_ch, hidden_channels, kernel_size, stride=stride, padding=padding
            )
            self.convs.append(conv)
            self.norms.append(modules.LayerNorm(hidden_channels))

        self.drop = nn.Dropout(0.1)
        self.gru = nn.GRU(
            hidden_channels,
            hidden_channels,
            n_gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        # Bidirectional -> 2 * hidden_channels
        self.proj = nn.Linear(hidden_channels * 2, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] where C = mel_channels
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            x = torch.relu(x)
            x = norm(x)
            x = self.drop(x)

        # Transpose for GRU: [B, T', C']
        x = x.transpose(1, 2)  # [B, T', hidden_channels]
        _, h_n = self.gru(x)  # h_n: [num_layers*2, B, hidden_channels]
        # Take the last forward and last backward states
        # h_n[-2] = last forward, h_n[-1] = last backward
        h_forward = h_n[-2]  # [B, hidden_channels]
        h_backward = h_n[-1]  # [B, hidden_channels]
        h = torch.cat([h_forward, h_backward], dim=-1)  # [B, hidden_channels*2]
        return self.proj(h).unsqueeze(-1)  # [B, out_channels, 1]


class TimbreEncoder(nn.Module):
    """Encodes voice identity (timbre) from a speaker ID or a reference mel.

    In speaker-ID mode (single/multi-speaker datasets), wraps a simple
    nn.Embedding. In reference mode, uses a ReferenceEncoder on a mel
    spectrogram clip so the same model can do zero-shot voice cloning at
    inference time.

    The output `g_timbre` conditions the Generator (dec) and the
    PosteriorEncoder (enc_q) — these are the parts of VITS that directly
    determine the acoustic quality / identity of the voice.
    """

    def __init__(
        self,
        n_speakers: int,
        gin_channels: int,
        ref_in_channels: int = 80,
        ref_hidden_channels: int = 256,
        ref_n_layers: int = 3,
        ref_kernel_size: int = 3,
        ref_stride: int = 2,
        ref_n_gru_layers: int = 1,
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
                n_gru_layers=ref_n_gru_layers,
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
    """Encodes prosodic / rhythmic / emotional information from a reference mel.

    The output `g_prosody` conditions the DurationPredictor (dp), i.e. it
    controls how long each phoneme lasts and therefore the overall rhythm
    and pacing of the utterance. It can optionally also inject into the
    normalizing flow to shape intonation contour.

    Accepts a reference mel spectrogram clip. For categorical emotion
    control, also accepts an emotion label (as a speaker-like ID embedding
    that projects into the same prosody space).
    """

    def __init__(
        self,
        out_channels: int,
        ref_in_channels: int = 80,
        ref_hidden_channels: int = 256,
        ref_n_layers: int = 3,
        ref_kernel_size: int = 3,
        ref_stride: int = 2,
        ref_n_gru_layers: int = 1,
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
            n_gru_layers=ref_n_gru_layers,
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
    """Encodes articulatory / pronunciation envelope information from a reference mel.

    The output `g_artic` conditions the normalizing flow (ResidualCouplingBlock)
    and, optionally, modulates the TextEncoder output. It captures how a
    given speaker realizes phonemes (e.g. accent patterns, coarticulation,
    vowel space shifts). Swapping this factor between speakers performs
    accent transfer while preserving the original voice timbre.
    """

    def __init__(
        self,
        out_channels: int,
        ref_in_channels: int = 80,
        ref_hidden_channels: int = 256,
        ref_n_layers: int = 3,
        ref_kernel_size: int = 3,
        ref_stride: int = 2,
        ref_n_gru_layers: int = 1,
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
            n_gru_layers=ref_n_gru_layers,
        )

    def forward(self, ref_mel: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        if ref_mel is not None:
            return self.ref_enc(ref_mel)
        return None


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training.

    Supports two operating modes:
      - **Legacy mode** (disentangled=False): single monolithic speaker
        embedding `g = emb_g(sid)` — identical to the original VITS
        behaviour and fully backward compatible with old checkpoints.
      - **Disentangled mode** (disentangled=True): three separate
        encoders (timbre, articulation, prosody) whose outputs are
        routed to different sub-modules, enabling independent control
        of voice identity, accent/pronunciation, and rhythm/emotion.

    In disentangled mode the conditioning signals are:
      * g_timbre   -> dec, enc_q
      * g_artic    -> flow
      * g_prosody  -> dp

    The flow additionally receives a concatenated "g_factors" tuple
    of (g_timbre, g_artic, g_prosody) projected through a per-flow
    linear layer so it can learn to modulate the latent with all
    three factors together.
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
        disentangled: bool = False,
        ref_enc_hidden_channels: int = 256,
        ref_enc_n_layers: int = 3,
        ref_enc_kernel_size: int = 3,
        ref_enc_stride: int = 2,
        ref_enc_n_gru_layers: int = 1,
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
        self.disentangled = disentangled

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

        if disentangled:
            self.timbre_enc = TimbreEncoder(
                n_speakers=n_speakers,
                gin_channels=gin_channels,
                ref_hidden_channels=ref_enc_hidden_channels,
                ref_n_layers=ref_enc_n_layers,
                ref_kernel_size=ref_enc_kernel_size,
                ref_stride=ref_enc_stride,
                ref_n_gru_layers=ref_enc_n_gru_layers,
                ref_enc_enabled=True,
            )
            self.artic_enc = ArticulationEncoder(
                out_channels=artic_dim if artic_dim > 0 else gin_channels,
                ref_hidden_channels=ref_enc_hidden_channels,
                ref_n_layers=ref_enc_n_layers,
                ref_kernel_size=ref_enc_kernel_size,
                ref_stride=ref_enc_stride,
                ref_n_gru_layers=ref_enc_n_gru_layers,
            )
            self.prosody_enc = ProsodyEncoder(
                out_channels=prosody_dim if prosody_dim > 0 else gin_channels,
                ref_hidden_channels=ref_enc_hidden_channels,
                ref_n_layers=ref_enc_n_layers,
                ref_kernel_size=ref_enc_kernel_size,
                ref_stride=ref_enc_stride,
                ref_n_gru_layers=ref_enc_n_gru_layers,
                n_emotion_labels=n_emotion_labels,
            )
            self._timbre_dim = timbre_dim if timbre_dim > 0 else gin_channels
            self._artic_dim = artic_dim if artic_dim > 0 else gin_channels
            self._prosody_dim = prosody_dim if prosody_dim > 0 else gin_channels

            # Optional projection for articulation onto the text encoder output.
            # In practice we apply it via a simple linear layer in the flow path
            # rather than modifying enc_p directly, keeping enc_p frozen-friendly
            # for LoRA fine-tuning scenarios.
            self.artic_proj = nn.Conv1d(
                self._artic_dim, hidden_channels, 1
            ) if self._artic_dim != hidden_channels else None

            # Concatenation projection for the flow conditioning:
            # the flow sees all three factors together.
            self.flow_cond_proj = nn.Conv1d(
                self._timbre_dim + self._artic_dim + self._prosody_dim,
                gin_channels,
                1,
            )
        else:
            if n_speakers > 1:
                self.emb_g = nn.Embedding(n_speakers, gin_channels)
            self.timbre_enc = None
            self.artic_enc = None
            self.prosody_enc = None

    def _get_g_legacy(self, sid):
        """Return the legacy monolithic speaker embedding."""
        if self.n_speakers > 1:
            return self.emb_g(sid).unsqueeze(-1)
        return None

    def _get_g_disentangled(self, sid, timbre_ref_mel, artic_ref_mel,
                            prosody_ref_mel, emotion_id):
        """Return (g_timbre, g_artic, g_prosody, g_flow) in disentangled mode."""
        g_timbre = self.timbre_enc(sid=sid, ref_mel=timbre_ref_mel)
        g_artic = self.artic_enc(ref_mel=artic_ref_mel)
        g_prosody = self.prosody_enc(ref_mel=prosody_ref_mel, emotion_id=emotion_id)

        # Ensure every factor has a tensor, falling back to zeros when missing.
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

        # Flow conditioning = concat of all three factors projected back to gin_channels.
        g_flow_input = torch.cat([g_timbre, g_artic, g_prosody], dim=1)
        g_flow = self.flow_cond_proj(g_flow_input)

        return g_timbre, g_artic, g_prosody, g_flow

    def forward(self, x, x_lengths, y, y_lengths, sid=None,
                timbre_ref_mel=None, artic_ref_mel=None,
                prosody_ref_mel=None, emotion_id=None):

        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        if self.disentangled:
            g_timbre, g_artic, g_prosody, g_flow = self._get_g_disentangled(
                sid, timbre_ref_mel, artic_ref_mel, prosody_ref_mel, emotion_id
            )
            # Articulation can modulate the text encoder prior (accent → phoneme realization)
            if self.artic_proj is not None:
                m_p = m_p + self.artic_proj(g_artic) * x_mask
            g = g_timbre  # legacy compatibility for dec / enc_q
        else:
            g = self._get_g_legacy(sid)
            g_timbre = g
            g_prosody = g
            g_flow = g

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_timbre)
        if self.disentangled:
            z_p = self.flow(z, y_mask, g=g_flow)
        else:
            z_p = self.flow(z, y_mask, g=g)

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

        if self.disentangled:
            g_timbre, g_artic, g_prosody, g_flow = self._get_g_disentangled(
                sid, timbre_ref_mel, artic_ref_mel, prosody_ref_mel, emotion_id
            )
            if self.artic_proj is not None:
                m_p = m_p + self.artic_proj(g_artic) * x_mask
        else:
            if self.n_speakers > 1:
                assert sid is not None, "Missing speaker id"
            g = self._get_g_legacy(sid)
            g_timbre = g
            g_prosody = g
            g_flow = g

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
        if self.disentangled:
            # In disentangled mode, voice conversion means swapping timbre
            # while keeping articulation and prosody optionally controllable.
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
        else:
            assert self.n_speakers > 1, "n_speakers have to be larger than 1."
            g_src = self.emb_g(sid_src).unsqueeze(-1)
            g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
            z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
            z_p = self.flow(z, y_mask, g=g_src)
            z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
            o_hat = self.dec(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)
