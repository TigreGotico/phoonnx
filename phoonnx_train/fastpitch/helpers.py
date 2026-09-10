"""
Alignment / masking helpers for the vendored ForwardTTS.

Adapted from coqui-ai/TTS ``TTS/tts/utils/helpers.py`` (© Coqui GmbH,
Mozilla Public License 2.0). Pure torch + numpy — the monotonic alignment
search (MAS) uses the numpy fallback implementation, so no cython/numba
build step is required.
"""
from typing import Optional

import numpy as np
import torch


def sequence_mask(sequence_length: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """[B] lengths -> [B, T_max] boolean mask."""
    if max_len is None:
        max_len = int(sequence_length.max().item())
    seq_range = torch.arange(max_len, dtype=sequence_length.dtype, device=sequence_length.device)
    return seq_range.unsqueeze(0) < sequence_length.unsqueeze(1)


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Expand per-token durations to a hard alignment path.

    duration: [B, T_en]
    mask:     [B, T_en, T_de]
    returns:  [B, T_en, T_de]
    """
    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, dim=1)
    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(path, (0, 0, 1, 0, 0, 0))[:, :-1]
    return path * mask


def average_over_durations(values: torch.Tensor, durs: torch.Tensor) -> torch.Tensor:
    """
    Average frame-level values over each token's duration span.

    values: [B, 1, T_de]
    durs:   [B, T_en]
    returns: [B, 1, T_en]
    """
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
    values_nonzero_cums = torch.nn.functional.pad(torch.cumsum(values != 0.0, dim=2), (1, 0))
    values_cums = torch.nn.functional.pad(torch.cumsum(values, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = values.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    values_sums = (torch.gather(values_cums, 2, dce) - torch.gather(values_cums, 2, dcs)).float()
    values_nelems = (torch.gather(values_nonzero_cums, 2, dce) - torch.gather(values_nonzero_cums, 2, dcs)).float()

    avg = torch.where(values_nelems == 0.0, values_nelems, values_sums / values_nelems)
    return avg.to(values.dtype)


def maximum_path(log_p: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Monotonic alignment search (numpy Viterbi), batch version.

    Port of coqui's ``maximum_path_numpy`` (TTS/tts/utils/helpers.py).

    log_p: [B, T_en, T_de] alignment log-probabilities
    mask:  [B, T_en, T_de]
    returns hard path [B, T_en, T_de]
    """
    max_neg_val = -np.inf
    value = (log_p * mask).detach().cpu().numpy().astype(np.float32)
    mask_np = mask.detach().cpu().numpy().astype(bool)

    b, t_x_max, t_y_max = value.shape
    paths = np.zeros_like(value, dtype=np.float32)
    for i in range(b):
        t_x = int(mask_np[i, :, 0].sum())
        t_y = int(mask_np[i, 0, :].sum())
        v = value[i, :t_x, :t_y]

        # forward pass
        direction = np.zeros((t_x, t_y), dtype=np.int32)
        score = np.full((t_x,), max_neg_val, dtype=np.float32)
        score[0] = v[0, 0]
        for y in range(1, t_y):
            new_score = np.full((t_x,), max_neg_val, dtype=np.float32)
            lo = max(0, t_x - (t_y - y))
            hi = min(t_x, y + 1)
            for x in range(lo, hi):
                stay = score[x]
                move = score[x - 1] if x > 0 else max_neg_val
                if move >= stay:
                    direction[x, y] = 1
                    new_score[x] = move + v[x, y]
                else:
                    new_score[x] = stay + v[x, y]
            score = new_score

        # backtrack
        x = t_x - 1
        for y in range(t_y - 1, -1, -1):
            paths[i, x, y] = 1.0
            if direction[x, y] == 1:
                x -= 1
    return torch.from_numpy(paths).to(device=log_p.device, dtype=log_p.dtype)
