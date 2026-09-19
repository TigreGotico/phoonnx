# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# BSD 3-Clause License
#
# Copyright (c) 2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#     this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# adapted from:
# https://github.com/NVIDIA/NeMo/blob/7256db10771aa1d213d9b49640667efaa14f89c9/nemo/collections/tts/parts/utils/helpers.py

from typing import Optional, Tuple

import numpy as np
import torch
from numba import jit, prange

# from nemo.collections.tts.torch.tts_data_types import DATA_STR2DATA_CLASS, MAIN_DATA_TYPES, WithLens
# from nemo.utils import logging
# from nemo.utils.decorators import deprecated

HAVE_WANDB = False
# try:
#     import wandb
# except ModuleNotFoundError:
#     HAVE_WANDB = False

# try:
#     from pytorch_lightning.utilities import rank_zero_only
# except ModuleNotFoundError:
#     from functools import wraps

    # def rank_zero_only(fn):
    #     @wraps(fn)
    #     def wrapped_fn(*args, **kwargs):
    #         logging.error(
    #             f"Function {fn} requires lighting to be installed, but it was not found. Please install lightning first"
    #         )
    #         exit(1)

def binarize_attention(attn, in_len, out_len):
    """Convert soft attention matrix to hard attention matrix.

    Args:
        attn (torch.Tensor): B x 1 x max_mel_len x max_text_len. Soft attention matrix.
        in_len (torch.Tensor): B. Lengths of texts.
        out_len (torch.Tensor): B. Lengths of spectrograms.

    Output:
        attn_out (torch.Tensor): B x 1 x max_mel_len x max_text_len. Hard attention matrix, final dim max_text_len should sum to 1.
    """
    b_size = attn.shape[0]
    with torch.no_grad():
        attn_cpu = attn.data.cpu().numpy()
        attn_out = torch.zeros_like(attn)
        for ind in range(b_size):
            hard_attn = mas(attn_cpu[ind, 0, : out_len[ind], : in_len[ind]])
            attn_out[ind, 0, : out_len[ind], : in_len[ind]] = torch.tensor(hard_attn, device=attn.device)
    return attn_out


def binarize_attention_parallel(attn, in_lens, out_lens):
    """For training purposes only. Binarizes attention with MAS.
           These will no longer receive a gradient.

        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
    with torch.no_grad():
        log_attn_cpu = torch.log(attn.data).cpu().numpy()
        attn_out = b_mas(log_attn_cpu, in_lens.cpu().numpy(), out_lens.cpu().numpy(), width=1)
    return torch.from_numpy(attn_out).to(attn.device)


def get_mask_from_lengths(lengths: Optional[torch.Tensor] = None, x: Optional[torch.Tensor] = None,) -> torch.Tensor:
    """Constructs binary mask from a 1D torch tensor of input lengths

    Args:
        lengths: Optional[torch.tensor] (torch.tensor): 1D tensor with lengths
        x: Optional[torch.tensor] = tensor to be used on, last dimension is for mask 
    Returns:
        mask (torch.tensor): num_sequences x max_length x 1 binary tensor
    """
    if lengths is None:
        assert x is not None
        return torch.ones(x.shape[-1], dtype=torch.bool, device=x.device)
    else:
        if x is None:
            max_len = torch.max(lengths)
        else:
            max_len = x.shape[-1]
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = ids < lengths.unsqueeze(1)
    return mask


def sort_tensor(
    context: torch.Tensor, lens: torch.Tensor, dim: Optional[int] = 0, descending: Optional[bool] = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sorts elements in context by the dim lengths specified in lens
    Args:
        context:  source tensor, sorted by lens
        lens: lengths of elements of context along the dimension dim
        dim: Optional[int] : dimension to sort by
    Returns:
        context: tensor sorted by lens along dimension dim
        lens_sorted: lens tensor, sorted
        ids_sorted: reorder ids to be used to restore original order
    
    """
    lens_sorted, ids_sorted = torch.sort(lens, descending=descending)
    context = torch.index_select(context, dim, ids_sorted)
    return context, lens_sorted, ids_sorted


def unsort_tensor(ordered: torch.Tensor, indices: torch.Tensor, dim: Optional[int] = 0) -> torch.Tensor:
    """Reverses the result of sort_tensor function:
       o, _, ids = sort_tensor(x,l) 
       assert unsort_tensor(o,ids) == x
    Args:
        ordered: context tensor, sorted by lengths
        indices: torch.tensor: 1D tensor with 're-order' indices returned by sort_tensor
    Returns:
        ordered tensor in original order (before calling sort_tensor)  
    """
    return torch.index_select(ordered, dim, indices.argsort(0))


@jit(nopython=True)
def mas(attn_map, width=1):
    # assumes mel x text
    opt = np.zeros_like(attn_map)
    attn_map = np.log(attn_map)
    attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]):  # for each text dim
            prev_j = np.arange(max(0, j - width), j + 1)
            prev_log = np.array([log_p[i - 1, prev_idx] for prev_idx in prev_j])

            ind = np.argmax(prev_log)
            log_p[i, j] = attn_map[i, j] + prev_log[ind]
            prev_ind[i, j] = prev_j[ind]

    # now backtrack
    curr_text_idx = attn_map.shape[1] - 1
    for i in range(attn_map.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1

    assert opt.sum(0).all()
    assert opt.sum(1).all()

    return opt


@jit(nopython=True)
def mas_width1(log_attn_map):
    """mas with hardcoded width=1"""
    # assumes mel x text
    neg_inf = log_attn_map.dtype.type(-np.inf)
    log_p = log_attn_map.copy()
    log_p[0, 1:] = neg_inf
    for i in range(1, log_p.shape[0]):
        prev_log1 = neg_inf
        for j in range(log_p.shape[1]):
            prev_log2 = log_p[i - 1, j]
            log_p[i, j] += max(prev_log1, prev_log2)
            prev_log1 = prev_log2

    # now backtrack
    opt = np.zeros_like(log_p)
    one = opt.dtype.type(1)
    j = log_p.shape[1] - 1
    for i in range(log_p.shape[0] - 1, 0, -1):
        opt[i, j] = one
        if log_p[i - 1, j - 1] >= log_p[i - 1, j]:
            j -= 1
            if j == 0:
                opt[1:i, j] = one
                break
    opt[0, j] = one
    return opt


@jit(nopython=True, parallel=True)
def b_mas(b_log_attn_map, in_lens, out_lens, width=1):
    assert width == 1
    attn_out = np.zeros_like(b_log_attn_map)

    for b in prange(b_log_attn_map.shape[0]):
        out = mas_width1(b_log_attn_map[b, 0, : out_lens[b], : in_lens[b]])
        attn_out[b, 0, : out_lens[b], : in_lens[b]] = out
    return attn_out



def regulate_len(
    durations, enc_out, pace=1.0, mel_max_len=None, group_size=1, dur_lens: torch.tensor = None,
):
    """A function that takes predicted durations per encoded token, and repeats enc_out according to the duration.
    NOTE: durations.shape[1] == enc_out.shape[1]

    Args:
        durations (torch.tensor): A tensor of shape (batch x enc_length) that represents how many times to repeat each
            token in enc_out.
        enc_out (torch.tensor): A tensor of shape (batch x enc_length x enc_hidden) that represents the encoded tokens.
        pace (float): The pace of speaker. Higher values result in faster speaking pace. Defaults to 1.0.        max_mel_len (int): The maximum length above which the output will be removed. If sum(durations, dim=1) >
            max_mel_len, the values after max_mel_len will be removed. Defaults to None, which has no max length.
        group_size (int): replicate the last element specified by durations[i, in_lens[i] - 1] until the
            full length of the sequence is the next nearest multiple of group_size
        in_lens (torch.tensor): input sequence length specifying valid values in the durations input tensor (only needed if group_size >1)
    """

    dtype = enc_out.dtype
    reps = durations.float() / pace
    reps = (reps + 0.5).floor().long()
    dec_lens = reps.sum(dim=1)
    if group_size > 1:
        to_pad = group_size * (torch.div(dec_lens + 1, group_size, rounding_mode='floor')) - dec_lens
        reps.index_put_(
            indices=[torch.arange(dur_lens.shape[0], dtype=torch.long), dur_lens - 1], values=to_pad, accumulate=True
        )
        dec_lens = reps.sum(dim=1)

    max_len = dec_lens.max()
    reps_cumsum = torch.cumsum(torch.nn.functional.pad(reps, (1, 0, 0, 0), value=0.0), dim=1)[:, None, :]
    reps_cumsum = reps_cumsum.to(dtype=dtype, device=enc_out.device)

    range_ = torch.arange(max_len).to(enc_out.device)[None, :, None]
    mult = (reps_cumsum[:, :, :-1] <= range_) & (reps_cumsum[:, :, 1:] > range_)
    mult = mult.to(dtype)
    enc_rep = torch.matmul(mult, enc_out)

    if mel_max_len is not None:
        enc_rep = enc_rep[:, :mel_max_len]
        dec_lens = torch.clamp_max(dec_lens, mel_max_len)

    return enc_rep, dec_lens


