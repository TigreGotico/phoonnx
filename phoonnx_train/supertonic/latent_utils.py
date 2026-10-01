"""Temporal latent compression and reference-crop helpers shared by the
text-to-latent module and the duration predictor.

The autoencoder produces a ``latent_dim``-channel latent at the spectrogram
frame rate. Both downstream stages operate on a *compressed* latent: groups of
``k`` consecutive frames are folded into the channel dimension, shrinking the
time axis ``k``-fold. Normalisation order matches the released model: normalise
the raw latent with the autoencoder's per-channel stats, then compress, then
apply the stage's own scalar multiplier.
"""
from __future__ import annotations

import torch

from phoonnx_train.supertonic.layers import make_mask


def compress(latent: torch.Tensor, k: int) -> torch.Tensor:
    """``(B, C, T)`` -> ``(B, C*k, T//k)``; trims T to a multiple of ``k``."""
    b, c, t = latent.shape
    t = (t // k) * k
    latent = latent[..., :t].view(b, c, t // k, k)
    return latent.permute(0, 1, 3, 2).reshape(b, c * k, t // k)


def decompress(latent: torch.Tensor, k: int, latent_dim: int) -> torch.Tensor:
    """Inverse of :func:`compress`: ``(B, C*k, T)`` -> ``(B, C, T*k)``."""
    b, ck, t = latent.shape
    latent = latent.view(b, latent_dim, k, t)
    return latent.permute(0, 1, 3, 2).reshape(b, latent_dim, t * k)


def normalize_and_compress(ae, raw: torch.Tensor, k: int, scale: float) -> torch.Tensor:
    return compress(ae.normalize(raw), k) * scale


def decompress_and_denormalize(ae, compressed: torch.Tensor, k: int, latent_dim: int, scale: float) -> torch.Tensor:
    return ae.denormalize(decompress(compressed / scale, k, latent_dim))


def sample_reference_crop(z1: torch.Tensor, lengths: torch.Tensor, frame_rate: float,
                          min_dur: float = 0.2, max_dur: float = 9.0):
    """Crop a random reference segment from each utterance's own compressed latent.

    Returns ``(ref_latent, ref_mask, ref_time_mask)`` where ``ref_time_mask``
    marks the positions of ``z1`` used as the reference (excluded from the flow
    matching loss to avoid leakage).
    """
    b, c, t = z1.shape
    dev = z1.device
    min_frames = max(1, round(min_dur * frame_rate))
    starts = torch.zeros(b, dtype=torch.long, device=dev)
    lens = torch.zeros(b, dtype=torch.long, device=dev)
    for i in range(b):
        li = int(lengths[i].item())
        top = max(min_frames, min(round(max_dur * frame_rate), li // 2))
        crop = min_frames if top <= min_frames else int(torch.randint(min_frames, top + 1, (1,)).item())
        crop = min(crop, max(li, 1))
        start = 0 if li - crop <= 0 else int(torch.randint(0, li - crop + 1, (1,)).item())
        starts[i], lens[i] = start, crop

    t_ref = int(lens.max().item())
    ref = z1.new_zeros(b, c, t_ref)
    ref_time_mask = z1.new_zeros(b, 1, t)
    for i in range(b):
        s, l = int(starts[i]), int(lens[i])
        ref[i, :, :l] = z1[i, :, s:s + l]
        ref_time_mask[i, :, s:s + l] = 1.0
    return ref, make_mask(lens, t_ref), ref_time_mask
