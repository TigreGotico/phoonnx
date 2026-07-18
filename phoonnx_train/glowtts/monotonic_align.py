"""
Monotonic Alignment Search (MAS) for GlowTTS.

MAS is the dynamic-programming algorithm from the GlowTTS paper (Kim et al.
2020, §3.2 / Algorithm 1) that finds the most probable monotonic hard
alignment between text tokens and mel frames under the current Gaussian
prior.

Two implementations are provided:

- The compiled Cython kernel already vendored for VITS at
  ``phoonnx_train/vits/monotonic_align`` (``maximum_path``) — VITS inherits
  the identical algorithm from GlowTTS. Used when the extension is built
  (``make -C phoonnx_train/vits/monotonic_align``), the fast path for real
  training runs.
- A pure-numpy fallback implementing the same DP (reconstructed from the
  paper's Algorithm 1), used automatically when the Cython extension is not
  compiled. Same results, just slower — fine for tests and small runs.
"""
import numpy as np

# torch (and the torch-importing Cython kernel wrapper) are deferred to
# maximum_path() so the pure-numpy DP stays importable in torch-free
# environments (e.g. for tests).


def _maximum_path_numpy(neg_cent: np.ndarray, t_ys: np.ndarray, t_xs: np.ndarray) -> np.ndarray:
    """
    Pure-numpy MAS DP (GlowTTS paper Algorithm 1).

    neg_cent: [b, t_y, t_x] log-likelihood of mel frame y under token x.
    Returns a hard 0/1 monotonic path of the same shape.
    """
    b = neg_cent.shape[0]
    paths = np.zeros_like(neg_cent, dtype=np.int32)
    max_neg = -1e9

    for i in range(b):
        t_y, t_x = int(t_ys[i]), int(t_xs[i])
        value = np.full((t_y, t_x), max_neg, dtype=np.float32)
        # forward pass
        for y in range(t_y):
            x_lo = max(0, t_x - (t_y - y))
            x_hi = min(t_x - 1, y)
            for x in range(x_lo, x_hi + 1):
                if y == 0:
                    v_prev = 0.0 if x == 0 else max_neg
                else:
                    v_cur = value[y - 1, x]
                    v_diag = value[y - 1, x - 1] if x > 0 else max_neg
                    v_prev = max(v_cur, v_diag)
                value[y, x] = neg_cent[i, y, x] + v_prev
        # backtrack
        x = t_x - 1
        for y in range(t_y - 1, -1, -1):
            paths[i, y, x] = 1
            if x > 0 and (x == y or value[y - 1, x - 1] >= value[y - 1, x]):
                x -= 1
    return paths


def maximum_path(neg_cent: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
    """
    Find the maximum-likelihood monotonic alignment path.

    Parameters
    ----------
    neg_cent : [b, t_mel, t_text] log-likelihood of each mel frame under
        each text token's Gaussian prior.
    mask : [b, t_mel, t_text] valid (non-padded) alignment mask.

    Returns
    -------
    [b, t_mel, t_text] hard 0/1 alignment path.
    """
    import torch

    try:
        from phoonnx_train.vits.monotonic_align import (
            maximum_path as _maximum_path_cython,
        )
    except ImportError:  # Cython extension not built
        _maximum_path_cython = None

    if _maximum_path_cython is not None:
        return _maximum_path_cython(neg_cent, mask)

    device, dtype = neg_cent.device, neg_cent.dtype
    neg_np = (neg_cent * mask).detach().cpu().numpy().astype(np.float32)
    t_ys = mask.sum(1)[:, 0].detach().cpu().numpy().astype(np.int32)
    t_xs = mask.sum(2)[:, 0].detach().cpu().numpy().astype(np.int32)
    path = _maximum_path_numpy(neg_np, t_ys, t_xs)
    return torch.from_numpy(path).to(device=device, dtype=dtype)
