"""Letter durations and the frame layout from monotonic alignment search.

Rápido predicts fractional letter durations and lays frames out per word: a
word gets a whole number of 25 Hz frames, and inside the word each letter sits
at a fractional position. One 25 Hz frame is often longer than a letter (a
stress mark or a combining tilde is a token of its own), so the search runs on
latent frames repeated ``upsample`` times. Every letter then gets at least
``1 / upsample`` of a frame, and the durations come out in frames with that
resolution.
"""
import math
from typing import Collection, List, Tuple

import numpy as np
import torch

from phoonnx.engines.rapido import frame_layout, word_layout
from phoonnx_train.matcha.mas import maximum_path


def gaussian_log_prior(mu: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Log N(z_t; mu_l, I) for every letter l and frame t: ``[B, L, T]``.

    mu: ``[B, L, D]`` prior means; z: ``[B, T, D]`` normalized latents.
    """
    d = mu.shape[-1]
    return (-0.5 * (z ** 2).sum(-1)[:, None, :]
            + mu @ z.transpose(1, 2)
            - 0.5 * (mu ** 2).sum(-1)[:, :, None]
            - 0.5 * d * math.log(2 * math.pi))


@torch.no_grad()
def align(mu: torch.Tensor, z: torch.Tensor, mask: torch.Tensor,
          fmask: torch.Tensor, upsample: int) -> torch.Tensor:
    """Monotonic alignment of letters to frames.

    Returns ``attn [B, L, T]``: the share of each frame that each letter holds
    (columns of real frames sum to 1). ``attn.sum(-1)`` is the letter
    duration in frames.
    """
    b, length, frames = mu.shape[0], mu.shape[1], z.shape[1]
    log_prior = gaussian_log_prior(mu.float(), z.float())
    log_prior = log_prior.repeat_interleave(upsample, dim=2)
    sub_mask = fmask.repeat_interleave(upsample, dim=1)
    path = maximum_path(log_prior, mask[:, :, None] & sub_mask[:, None, :])
    return path.view(b, length, frames, upsample).mean(-1)


def aligned_word_frames(durations: np.ndarray, cw: np.ndarray, n_frames: int) -> np.ndarray:
    """Whole frames per word from fractional letter durations.

    Word boundaries are the cumulative durations rounded to the nearest
    frame, so the counts always sum to ``n_frames``. Every word keeps at least
    one frame when there are enough frames, as at inference, where word
    counts are clamped to at least one.
    """
    n_words = int(cw.max()) + 1
    word_dur = np.bincount(cw, weights=durations, minlength=n_words)
    ends = np.rint(np.cumsum(word_dur)).astype(np.int64)
    ends[-1] = n_frames
    if n_frames >= n_words:
        k = np.arange(n_words)
        ends = np.clip(ends, k + 1, n_frames - (n_words - 1 - k))
        # ends - k non-decreasing makes ends strictly increasing
        ends = np.maximum.accumulate(ends - k) + k
    return np.diff(np.concatenate([[0], ends]))


def batch_layout(ids: torch.Tensor, mask: torch.Tensor, durations: torch.Tensor,
                 fmask: torch.Tensor, boundary_ids: Collection[int],
                 ) -> Tuple[torch.Tensor, ...]:
    """Padded word and frame layout of a batch, from letter durations.

    Returns ``cw, wstart`` ``[B, L]`` (``cw = -1`` on padding) and
    ``fw, fp`` ``[B, T]`` (zero on padding).
    """
    b, length = ids.shape
    frames = fmask.shape[1]
    cw = torch.full((b, length), -1, dtype=torch.long)
    wstart = torch.zeros((b, length), dtype=torch.long)
    fw = torch.zeros((b, frames), dtype=torch.long)
    fp = torch.zeros((b, frames), dtype=torch.float32)
    lengths: List[int] = mask.sum(1).tolist()
    n_frames: List[int] = fmask.sum(1).tolist()
    for i in range(b):
        row_cw, row_ws = word_layout(ids[i, :lengths[i]].tolist(), boundary_ids)
        counts = aligned_word_frames(
            durations[i, :lengths[i]].detach().double().cpu().numpy(), row_cw, n_frames[i])
        row_fw, row_fp = frame_layout(counts)
        cw[i, :lengths[i]] = torch.from_numpy(row_cw)
        wstart[i, :lengths[i]] = torch.from_numpy(row_ws)
        fw[i, :n_frames[i]] = torch.from_numpy(row_fw)
        fp[i, :n_frames[i]] = torch.from_numpy(row_fp)
    device = ids.device
    return cw.to(device), wstart.to(device), fw.to(device), fp.to(device)
