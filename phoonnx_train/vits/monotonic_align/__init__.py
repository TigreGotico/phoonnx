import numpy as np
import torch


def maximum_path(neg_cent, mask):
    """Monotonic alignment search, vectorized over the batch in numpy.

    neg_cent: [b, t_t, t_s]
    mask: [b, t_t, t_s]
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    # [b, t_s, t_t]: rows = text positions, columns = spectrogram frames
    value = neg_cent.detach().cpu().numpy().astype(np.float32).transpose(0, 2, 1)
    mask_np = mask.detach().cpu().numpy().astype(bool).transpose(0, 2, 1)
    value *= mask_np

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
    direction = np.where(mask_np, direction, 1)

    path = np.zeros(value.shape, dtype=np.float32)
    index = mask_np[:, :, 0].sum(1).astype(np.int64) - 1
    index_range = np.arange(b)
    for j in reversed(range(t_t)):
        path[index_range, index, j] = 1
        index = np.maximum(0, index + direction[index_range, index, j] - 1)
    path *= mask_np
    return torch.from_numpy(path.transpose(0, 2, 1)).to(device=device, dtype=dtype)
