"""Pure-numpy monotonic alignment for the vendored StyleTTS2 trainer.

Replaces upstream's compiled ``monotonic_align`` Cython extension so the
training package imports (and unit-tests) without a build step. The DP is
identical; it only runs during stage-1 alignment, where the extra Python
cost is dwarfed by the model forward/backward.
"""
import numpy as np
import torch


def _maximum_path_each(value: np.ndarray, t_x: int, t_y: int) -> np.ndarray:
    """Monotonic Viterbi path for one (t_x, t_y) score matrix."""
    path = np.zeros_like(value, dtype=np.int32)
    v = np.full((t_x, t_y), -np.inf, dtype=np.float32)
    v[0, 0] = value[0, 0]
    for y in range(1, t_y):
        v[0, y] = v[0, y - 1] + value[0, y]
    for x in range(1, t_x):
        for y in range(x, t_y):
            stay = v[x, y - 1] if y > x else -np.inf
            move = v[x - 1, y - 1]
            v[x, y] = max(stay, move) + value[x, y]
    x = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[x, y] = 1
        if x > 0 and (y == x or v[x, y - 1] < v[x - 1, y - 1]):
            x -= 1
    return path


def maximum_path(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Batched monotonic alignment path.

    :param value: [B, T_x, T_y] alignment scores
    :param mask: [B, T_x, T_y] validity mask
    :return: [B, T_x, T_y] hard monotonic path (same dtype/device as value)
    """
    device, dtype = value.device, value.dtype
    val = (value * mask).detach().cpu().numpy().astype(np.float32)
    msk = mask.detach().cpu().numpy()
    t_xs = msk.sum(1)[:, 0].astype(np.int32)
    t_ys = msk.sum(2)[:, 0].astype(np.int32)
    out = np.zeros_like(val, dtype=np.int32)
    for b in range(val.shape[0]):
        t_x, t_y = int(t_xs[b]), int(t_ys[b])
        if t_x > 0 and t_y > 0:
            out[b, :t_x, :t_y] = _maximum_path_each(val[b, :t_x, :t_y], t_x, t_y)
    return torch.from_numpy(out).to(device=device, dtype=dtype)


def mask_from_lens(sim: torch.Tensor, in_lens: torch.Tensor, out_lens: torch.Tensor) -> torch.Tensor:
    """Validity mask [B, T_in, T_out] from per-utterance lengths."""
    b, t_in, t_out = sim.shape
    in_mask = torch.arange(t_in, device=sim.device)[None, :] < in_lens[:, None]
    out_mask = torch.arange(t_out, device=sim.device)[None, :] < out_lens[:, None]
    return (in_mask[:, :, None] & out_mask[:, None, :]).to(sim.dtype)
