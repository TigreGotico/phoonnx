"""Speech autoencoder: waveform -> log-spectrogram -> continuous latent ->
waveform. A Vocos-style ConvNeXt encoder compresses the spectrogram to a
low-dimensional latent; a causal ConvNeXt decoder with a WaveNeXt-style head
projects each latent frame straight into ``hop_length`` waveform samples (no
transposed convolutions or ISTFT).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from phoonnx_train.supertonic.config import AutoencoderConfig
from phoonnx_train.supertonic.layers import ConvNeXtBlock


class SpecFront(nn.Module):
    """Log mel-spectrogram, optionally concatenated with the log linear spectrum."""

    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        self.cfg = cfg
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, win_length=cfg.win_length,
            hop_length=cfg.hop_length, n_mels=cfg.n_mels, power=1.0, center=True,
        )
        self.lin = torchaudio.transforms.Spectrogram(
            n_fft=cfg.n_fft, win_length=cfg.win_length, hop_length=cfg.hop_length,
            power=1.0, center=True,
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        mel = torch.log(self.mel(wav).clamp_min(1e-5))
        if not self.cfg.concat_linear_spec:
            return mel
        lin = torch.log(self.lin(wav).clamp_min(1e-5))
        return torch.cat([mel, lin], dim=1)


class LatentEncoder(nn.Module):
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        self.spec = SpecFront(cfg)
        self.pad = cfg.encoder_kernel // 2
        self.proj_in = nn.Conv1d(cfg.input_dim, cfg.hidden_dim, cfg.encoder_kernel)
        self.bn = nn.BatchNorm1d(cfg.hidden_dim)
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock(cfg.hidden_dim, cfg.ffn_dim, cfg.encoder_kernel) for _ in range(cfg.encoder_layers)]
        )
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.proj_out = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = F.pad(self.spec(wav), (self.pad, self.pad), mode="replicate")
        x = self.bn(self.proj_in(x))
        for block in self.blocks:
            x = block(x)
        x = self.proj_out(self.norm(x.transpose(1, 2)))
        return x.transpose(1, 2)  # (B, latent_dim, T)


class LatentDecoder(nn.Module):
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        self.cfg = cfg
        self.k = cfg.decoder_kernel
        self.proj_in = nn.Conv1d(cfg.latent_dim, cfg.hidden_dim, cfg.decoder_kernel)
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock(cfg.hidden_dim, cfg.ffn_dim, cfg.decoder_kernel, dilation=d, causal=True)
             for d in cfg.decoder_dilations]
        )
        self.bn = nn.BatchNorm1d(cfg.hidden_dim)
        self.head_conv = nn.Conv1d(cfg.hidden_dim, cfg.head_dim, 3)
        self.head_act = nn.PReLU()
        self.head_proj = nn.Linear(cfg.head_dim, cfg.hop_length, bias=False)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = F.pad(latent, (self.k - 1, 0), mode="replicate")
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.bn(x)
        x = F.pad(x, (2, 0), mode="replicate")
        x = self.head_act(self.head_conv(x))
        x = self.head_proj(x.transpose(1, 2))  # (B, T, hop)
        return x.reshape(x.shape[0], -1)


class SpeechAutoencoder(nn.Module):
    """Owns per-channel latent statistics shared with the two downstream stages."""

    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = LatentEncoder(cfg)
        self.decoder = LatentDecoder(cfg)
        self.register_buffer("latent_mean", torch.zeros(1, cfg.latent_dim, 1))
        self.register_buffer("latent_std", torch.ones(1, cfg.latent_dim, 1))
        self.register_buffer("stats_fitted", torch.tensor(False))

    def forward(self, wav: torch.Tensor):
        latent = self.encoder(wav)
        recon = self.decoder(latent)
        n = min(wav.shape[-1], recon.shape[-1])
        return recon[..., :n], latent

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        return self.encoder(wav)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    @torch.no_grad()
    def fit_latent_stats(self, latents) -> None:
        flat = torch.cat([lat.transpose(0, 1).reshape(lat.shape[1], -1) for lat in latents], dim=1)
        self.latent_mean.copy_(flat.mean(dim=1).view(1, -1, 1))
        self.latent_std.copy_(flat.std(dim=1).clamp_min(1e-5).view(1, -1, 1))
        self.stats_fitted.fill_(True)

    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.latent_mean) / self.latent_std

    def denormalize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * self.latent_std + self.latent_mean
