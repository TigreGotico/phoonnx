import numpy as np
import torch

from .core_numpy import maximum_path_numpy


def maximum_path(neg_cent, mask):
    """Monotonic alignment search.

    neg_cent: [b, t_t, t_s]
    mask: [b, t_t, t_s]
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    # core works on [b, t_s, t_t]: rows = text positions, columns = frames
    value = neg_cent.detach().cpu().numpy().astype(np.float32).transpose(0, 2, 1)
    mask_np = mask.detach().cpu().numpy().astype(bool).transpose(0, 2, 1)
    path = maximum_path_numpy(value, mask_np)
    return torch.from_numpy(path.transpose(0, 2, 1)).to(device=device, dtype=dtype)
