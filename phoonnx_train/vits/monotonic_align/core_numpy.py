import numpy as np


def maximum_path_numpy(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Monotonic alignment search, vectorized over the batch.

    value: [b, t_s, t_t] float32 — rows = text positions, columns = frames
    mask: [b, t_s, t_t] bool
    Returns the alignment path [b, t_s, t_t] float32.
    """
    value = value * mask
    b, t_s, t_t = value.shape
    max_neg_val = -1e9
    direction = np.zeros(value.shape, dtype=np.int64)
    v = np.zeros((b, t_s), dtype=np.float32)
    s_range = np.arange(t_s).reshape(1, -1)
    for j in range(t_t):
        v0 = np.pad(v, [[0, 0], [1, 0]], mode="constant", constant_values=max_neg_val)[:, :-1]
        max_mask = v >= v0
        v_max = np.where(max_mask, v, v0)
        direction[:, :, j] = max_mask
        v = np.where(s_range <= j, v_max + value[:, :, j], max_neg_val)
    direction = np.where(mask, direction, 1)

    path = np.zeros(value.shape, dtype=np.float32)
    index = mask[:, :, 0].sum(1).astype(np.int64) - 1
    index_range = np.arange(b)
    for j in reversed(range(t_t)):
        path[index_range, index, j] = 1
        index = np.maximum(0, index + direction[index_range, index, j] - 1)
    return path * mask
