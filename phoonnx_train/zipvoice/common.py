"""Model-side helpers from ZipVoice ``zipvoice/utils/common.py`` (the subset
the model and trainer need; the k2-fsa original also carries recipe/CLI
utilities that do not apply here)."""
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """2-D bool tensor: True at padded positions, False at valid ones."""
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)


def prepare_avg_tokens_durations(features_lens, tokens_lens):
    tokens_durations = []
    for i in range(len(features_lens)):
        utt_duration = features_lens[i]
        avg_token_duration = utt_duration // tokens_lens[i]
        tokens_durations.append([avg_token_duration] * tokens_lens[i])
    return tokens_durations


def pad_labels(y: List[List[int]], pad_id: int, device: torch.device):
    """Pad the transcripts to the same length with the pad id (every
    sequence also gets one trailing pad, as upstream)."""
    y = [token_ids + [pad_id] for token_ids in y]
    length = max([len(token_ids) for token_ids in y])
    y = [token_ids + [pad_id] * (length - len(token_ids)) for token_ids in y]
    return torch.tensor(y, dtype=torch.int64, device=device)


def get_tokens_index(durations: List[List[int]], num_frames: int) -> torch.Tensor:
    """Position in the transcript for each frame, shape (batch, num_frames)."""
    durations = [x + [num_frames - sum(x)] for x in durations]
    batch_size = len(durations)
    ans = torch.zeros(batch_size, num_frames, dtype=torch.int64)
    for b in range(batch_size):
        this_dur = durations[b]
        cur_frame = 0
        for i, d in enumerate(this_dur):
            ans[b, cur_frame : cur_frame + d] = i
            cur_frame += d
        assert cur_frame == num_frames, (cur_frame, num_frames)
    return ans


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    """Propagate the (duration-adjusted) batch count into every submodule
    that schedules on it (Zipformer's dropout/whitening schedules)."""
    if isinstance(model, DDP):
        model = model.module
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name


def condition_time_mask(
    features_lens: torch.Tensor,
    mask_percent: Tuple[float, float],
    max_len: int = 0,
) -> torch.Tensor:
    """Random contiguous time mask covering ``mask_percent`` of each item:
    True on the masked (generation-target) region."""
    mask_size = (
        torch.zeros_like(features_lens, dtype=torch.float32).uniform_(*mask_percent)
        * features_lens
    ).to(torch.int64)
    mask_starts = (
        torch.rand_like(mask_size, dtype=torch.float32) * (features_lens - mask_size)
    ).to(torch.int64)
    mask_ends = mask_starts + mask_size
    max_len = max(max_len, features_lens.max())
    seq_range = torch.arange(0, max_len, device=features_lens.device)
    mask = (seq_range[None, :] >= mask_starts[:, None]) & (
        seq_range[None, :] < mask_ends[:, None]
    )
    return mask
