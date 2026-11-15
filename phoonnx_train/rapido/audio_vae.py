"""VoxCPM2 AudioVAE v2: 16 kHz audio <-> 64-dim latents at 25 Hz <-> 48 kHz audio.

A causal convolutional VAE with Snake activations and weight-normalised convolutions.
The encoder downsamples 16 kHz audio by 640 (rates 2, 5, 8, 8); the decoder upsamples
by 1920 (rates 8, 6, 5, 2, 2, 2) to 48 kHz, with a per-block scale and bias chosen by
the output sample-rate bucket. Rápido keeps it frozen.

Parameter names match ``audiovae.pth`` in ``openbmb/VoxCPM2`` (Apache 2.0), weight
norm included (``weight_g`` / ``weight_v``), so ``load_state_dict(strict=True)`` loads
it. :meth:`AudioVAE.remove_weight_norm` folds the norm into plain weights for export.
"""
import bisect
import json
import math
from dataclasses import dataclass, fields
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ID = "openbmb/VoxCPM2"
REVISION = "7936080b41163874f38610f46f438e2c0757f9fd"
WEIGHTS_FILE = "audiovae.pth"
CONFIG_FILE = "config.json"


@dataclass(frozen=True)
class AudioVAEConfig:
    """The ``audio_vae_config`` block of VoxCPM2's ``config.json``."""

    encoder_dim: int = 128
    encoder_rates: Tuple[int, ...] = (2, 5, 8, 8)
    latent_dim: int = 64
    decoder_dim: int = 2048
    decoder_rates: Tuple[int, ...] = (8, 6, 5, 2, 2, 2)
    depthwise: bool = True
    sample_rate: int = 16000
    out_sample_rate: int = 48000
    sr_bin_boundaries: Tuple[int, ...] = (20000, 30000, 40000)

    @classmethod
    def from_dict(cls, cfg: dict) -> "AudioVAEConfig":
        names = {f.name for f in fields(cls)}
        unknown = {k: v for k, v in cfg.items() if k not in names}
        # the released model uses the defaults for every option this module leaves out
        supported = {"use_noise_block": False, "cond_type": "scale_bias", "cond_out_layer": False, "cond_dim": 128}
        if any(supported.get(k, object()) != v for k, v in unknown.items()):
            raise ValueError(f"unsupported AudioVAE options: {unknown}")
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items() if k in names})

    @property
    def hop_length(self) -> int:
        """Input samples per latent frame."""
        return math.prod(self.encoder_rates)

    @property
    def decode_hop_length(self) -> int:
        """Output samples per latent frame."""
        return math.prod(self.decoder_rates)


class _WeightNormConv(nn.Module):
    """A convolution whose weight is ``weight_g * weight_v / ||weight_v||`` (norm over
    every dim but 0), until :meth:`remove_weight_norm` folds it into ``weight``."""

    def __init__(self, weight_shape: Tuple[int, ...], out_channels: int, bias: bool):
        super().__init__()
        v = torch.empty(weight_shape)
        nn.init.trunc_normal_(v, std=0.02)
        self.weight_v = nn.Parameter(v)
        self.weight_g = nn.Parameter(self._norm(v))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    @staticmethod
    def _norm(v: torch.Tensor) -> torch.Tensor:
        return v.norm(dim=tuple(range(1, v.dim())), keepdim=True)

    def _weight(self) -> torch.Tensor:
        if "weight" in self._parameters:
            return self.weight
        return self.weight_v * (self.weight_g / self._norm(self.weight_v))

    def remove_weight_norm(self) -> None:
        weight = self._weight().detach()
        del self.weight_g, self.weight_v
        self.weight = nn.Parameter(weight)


class CausalConv1d(_WeightNormConv):
    """Weight-normalised Conv1d padded on the left by ``2 * padding - output_padding``."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 dilation: int = 1, groups: int = 1, padding: int = 0, output_padding: int = 0,
                 bias: bool = True):
        super().__init__((out_channels, in_channels // groups, kernel_size), out_channels, bias)
        self.stride, self.dilation, self.groups = stride, dilation, groups
        self.left_pad = padding * 2 - output_padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, 0))
        return F.conv1d(x, self._weight(), self.bias, self.stride, 0, self.dilation, self.groups)


class CausalConvTranspose1d(_WeightNormConv):
    """Weight-normalised ConvTranspose1d trimmed on the right by ``2 * padding - output_padding``."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int,
                 padding: int, output_padding: int):
        super().__init__((in_channels, out_channels, kernel_size), out_channels, bias=True)
        self.stride = stride
        self.trim = padding * 2 - output_padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose1d(x, self._weight(), self.bias, self.stride)[..., :-self.trim]


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int, groups: int):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(dim),
            CausalConv1d(dim, dim, 7, dilation=dilation, groups=groups, padding=3 * dilation),
            Snake1d(dim),
            CausalConv1d(dim, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


def _residual_stack(dim: int, groups: int) -> List[nn.Module]:
    return [ResidualUnit(dim, dilation, groups) for dilation in (1, 3, 9)]


class EncoderBlock(nn.Module):
    def __init__(self, out_dim: int, stride: int, groups: int):
        super().__init__()
        in_dim = out_dim // 2
        self.block = nn.Sequential(
            *_residual_stack(in_dim, groups), Snake1d(in_dim),
            CausalConv1d(in_dim, out_dim, 2 * stride, stride=stride, padding=math.ceil(stride / 2),
                         output_padding=stride % 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int, groups: int):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(in_dim),
            CausalConvTranspose1d(in_dim, out_dim, 2 * stride, stride, padding=math.ceil(stride / 2),
                                  output_padding=stride % 2),
            *_residual_stack(out_dim, groups))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    """16 kHz audio [B,1,N] -> posterior mean and log-variance [B,latent,N/640]."""

    def __init__(self, config: AudioVAEConfig):
        super().__init__()
        dim = config.encoder_dim
        layers: List[nn.Module] = [CausalConv1d(1, dim, 7, padding=3)]
        for stride in config.encoder_rates:
            dim *= 2
            layers.append(EncoderBlock(dim, stride, dim // 2 if config.depthwise else 1))
        self.block = nn.Sequential(*layers)
        self.fc_mu = CausalConv1d(dim, config.latent_dim, 3, padding=1)
        self.fc_logvar = CausalConv1d(dim, config.latent_dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.block(x)
        return self.fc_mu(hidden), self.fc_logvar(hidden)


class SampleRateCondition(nn.Module):
    """Per-channel scale and bias chosen by the output sample-rate bucket."""

    def __init__(self, channels: int, buckets: int):
        super().__init__()
        self.scale_embed = nn.Embedding(buckets, channels)
        self.bias_embed = nn.Embedding(buckets, channels)
        nn.init.ones_(self.scale_embed.weight)
        nn.init.zeros_(self.bias_embed.weight)

    def forward(self, x: torch.Tensor, bucket: torch.Tensor) -> torch.Tensor:
        return x * self.scale_embed(bucket).unsqueeze(-1) + self.bias_embed(bucket).unsqueeze(-1)


class Decoder(nn.Module):
    """Latents [B,latent,T] -> audio [B,1,T*1920] in [-1, 1]."""

    def __init__(self, config: AudioVAEConfig):
        super().__init__()
        latent, channels = config.latent_dim, config.decoder_dim
        if config.depthwise:
            layers: List[nn.Module] = [CausalConv1d(latent, latent, 7, groups=latent, padding=3),
                                       CausalConv1d(latent, channels, 1)]
        else:
            layers = [CausalConv1d(latent, channels, 7, padding=3)]
        first_block = len(layers)
        for i, stride in enumerate(config.decoder_rates):
            in_dim, out_dim = channels // 2 ** i, channels // 2 ** (i + 1)
            layers.append(DecoderBlock(in_dim, out_dim, stride, out_dim if config.depthwise else 1))
        layers += [Snake1d(out_dim), CausalConv1d(out_dim, 1, 7, padding=3), nn.Tanh()]
        self.model = nn.ModuleList(layers)

        self.register_buffer("sr_bin_boundaries", torch.tensor(config.sr_bin_boundaries, dtype=torch.int32))
        buckets = len(config.sr_bin_boundaries) + 1
        self.sr_cond_model = nn.ModuleList(
            SampleRateCondition(channels // 2 ** (i - first_block), buckets)
            if first_block <= i < first_block + len(config.decoder_rates) else None
            for i in range(len(layers)))

    def bucket(self, sample_rate: int) -> int:
        """The sample-rate bucket index (``torch.bucketize`` with right=False)."""
        return bisect.bisect_left(self.sr_bin_boundaries.tolist(), sample_rate)

    def forward(self, z: torch.Tensor, sample_rate: int) -> torch.Tensor:
        bucket = torch.full((z.shape[0],), self.bucket(sample_rate), dtype=torch.long, device=z.device)
        x = z
        for layer, condition in zip(self.model, self.sr_cond_model):
            if condition is not None:
                x = condition(x, bucket)
            x = layer(x)
        return x


class AudioVAE(nn.Module):
    def __init__(self, config: AudioVAEConfig = AudioVAEConfig()):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def encode(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Audio [B,1,N] or [B,N] -> latent means [B,latent,ceil(N/640)].

        ``sample_rate`` must be ``config.sample_rate`` (16 kHz); resample before calling.
        The audio is right-padded with zeros to a whole number of frames.
        """
        if sample_rate != self.config.sample_rate:
            raise ValueError(f"AudioVAE encodes {self.config.sample_rate} Hz audio, got {sample_rate} Hz")
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        hop = self.config.hop_length
        audio = F.pad(audio, (0, math.ceil(audio.shape[-1] / hop) * hop - audio.shape[-1]))
        return self.encoder(audio)[0]

    def decode(self, z: torch.Tensor, sample_rate: Optional[int] = None) -> torch.Tensor:
        """Latents [B,latent,T] -> audio [B,1,T*1920] at ``config.out_sample_rate``."""
        return self.decoder(z, sample_rate or self.config.out_sample_rate)

    def remove_weight_norm(self) -> None:
        for module in self.modules():
            if isinstance(module, _WeightNormConv) and "weight_v" in module._parameters:
                module.remove_weight_norm()


def load_audio_vae(device: str = "cpu") -> AudioVAE:
    """The frozen VoxCPM2 AudioVAE v2 from the Hub, weights loaded strictly, in eval mode."""
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(REPO_ID, CONFIG_FILE, revision=REVISION), encoding="utf-8") as f:
        config = AudioVAEConfig.from_dict(json.load(f)["audio_vae_config"])
    vae = AudioVAE(config)
    state = torch.load(hf_hub_download(REPO_ID, WEIGHTS_FILE, revision=REVISION), map_location="cpu", weights_only=True)
    vae.load_state_dict(state["state_dict"], strict=True)
    return vae.to(device).eval().requires_grad_(False)
