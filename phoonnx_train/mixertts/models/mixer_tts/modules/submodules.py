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

# adapted from:
# https://github.com/NVIDIA/NeMo/blob/7256db10771aa1d213d9b49640667efaa14f89c9/nemo/collections/tts/modules/submodules.py

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

# from nemo.core.classes import NeuralModule, adapter_mixins
# from nemo.core.neural_types.elements import EncodedRepresentation, Index, LengthsType, MelSpectrogramType
# from nemo.core.neural_types.neural_type import NeuralType
# from nemo.utils import logging


SUPPORTED_CONDITION_TYPES = ["add", "concat", "layernorm"]


def check_support_condition_types(condition_types):
    for tp in condition_types:
        if tp not in SUPPORTED_CONDITION_TYPES:
            raise ValueError(f"Unknown conditioning type {tp}")


def masked_instance_norm(
    input: Tensor, mask: Tensor, weight: Tensor, bias: Tensor, momentum: float, eps: float = 1e-5,
) -> Tensor:
    r"""Applies Masked Instance Normalization for each channel in each data sample in a batch.

    See :class:`~MaskedInstanceNorm1d` for details.
    """
    lengths = mask.sum((-1,))
    mean = (input * mask).sum((-1,)) / lengths  # (N, C)
    var = (((input - mean[(..., None)]) * mask) ** 2).sum((-1,)) / lengths  # (N, C)
    out = (input - mean[(..., None)]) / torch.sqrt(var[(..., None)] + eps)  # (N, C, ...)
    out = out * weight[None, :][(..., None)] + bias[None, :][(..., None)]

    return out


class MaskedInstanceNorm1d(torch.nn.InstanceNorm1d):
    r"""Applies Instance Normalization over a masked 3D input
    (a mini-batch of 1D inputs with additional channel dimension)..

    See documentation of :class:`~torch.nn.InstanceNorm1d` for details.

    Shape:
        - Input: :math:`(N, C, L)`
        - Mask: :math:`(N, 1, L)`
        - Output: :math:`(N, C, L)` (same shape as input)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ) -> None:
        super(MaskedInstanceNorm1d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input: Tensor, mask: Tensor) -> Tensor:
        return masked_instance_norm(input, mask, self.weight, self.bias, self.momentum, self.eps,)


class PartialConv1d(torch.nn.Conv1d):
    """
    Zero padding creates a unique identifier for where the edge of the data is, such that the model can almost always identify
    exactly where it is relative to either edge given a sufficient receptive field. Partial padding goes to some lengths to remove 
    this affect.
    """

    __constants__ = ['slide_winsize']
    slide_winsize: float

    def __init__(self, *args, **kwargs):
        super(PartialConv1d, self).__init__(*args, **kwargs)
        weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0])
        self.register_buffer("weight_maskUpdater", weight_maskUpdater, persistent=False)
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2]

    def forward(self, input, mask_in):
        if mask_in is None:
            mask = torch.ones(1, 1, input.shape[2], dtype=input.dtype, device=input.device)
        else:
            mask = mask_in
            input = torch.mul(input, mask)
        with torch.no_grad():
            update_mask = F.conv1d(
                mask,
                self.weight_maskUpdater,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=1,
            )
            update_mask_filled = torch.masked_fill(update_mask, update_mask == 0, self.slide_winsize)
            mask_ratio = self.slide_winsize / update_mask_filled
            update_mask = torch.clamp(update_mask, 0, 1)
            mask_ratio = torch.mul(mask_ratio, update_mask)

        raw_out = self._conv_forward(input, self.weight, self.bias)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1)
            output = torch.mul(raw_out - bias_view, mask_ratio) + bias_view
            output = torch.mul(output, update_mask)
        else:
            output = torch.mul(raw_out, mask_ratio)

        return output


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(nn.Module):
    __constants__ = ['use_partial_padding']
    use_partial_padding: bool

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain='linear',
        use_partial_padding=False,
        use_weight_norm=False,
        norm_fn=None,
    ):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.use_partial_padding = use_partial_padding
        conv_fn = torch.nn.Conv1d
        if use_partial_padding:
            conv_fn = PartialConv1d
        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
        if use_weight_norm:
            self.conv = torch.nn.utils.weight_norm(self.conv)
        if norm_fn is not None:
            self.norm = norm_fn(out_channels, affine=True)
        else:
            self.norm = None

    def forward(self, signal, mask=None):
        if self.use_partial_padding:
            ret = self.conv(signal, mask)
            if self.norm is not None:
                ret = self.norm(ret, mask)
        else:
            if mask is not None:
                signal = signal.mul(mask)
            ret = self.conv(signal)
            if self.norm is not None:
                ret = self.norm(ret)

        # if self.is_adapter_available():
        #     ret = self.forward_enabled_adapters(ret.transpose(1, 2)).transpose(1, 2)

        return ret



class ConditionalLayerNorm(torch.nn.LayerNorm):
    """
    This module is used to condition torch.nn.LayerNorm.
    If we don't have any conditions, this will be a normal LayerNorm.
    """

    def __init__(self, hidden_dim, condition_dim=None, condition_types=[]):
        check_support_condition_types(condition_types)
        self.condition = "layernorm" in condition_types
        super().__init__(hidden_dim, elementwise_affine=not self.condition)

        if self.condition:
            self.cond_weight = torch.nn.Linear(condition_dim, hidden_dim)
            self.cond_bias = torch.nn.Linear(condition_dim, hidden_dim)
            self.init_parameters()

    def init_parameters(self):
        torch.nn.init.constant_(self.cond_weight.weight, 0.0)
        torch.nn.init.constant_(self.cond_weight.bias, 1.0)
        torch.nn.init.constant_(self.cond_bias.weight, 0.0)
        torch.nn.init.constant_(self.cond_bias.bias, 0.0)

    def forward(self, inputs, conditioning=None):
        inputs = super().forward(inputs)

        # Normalize along channel
        if self.condition:
            if conditioning is None:
                raise ValueError(
                    """You should add additional data types as conditions (e.g. speaker id or reference audio) 
                                 and define speaker_encoder in your config."""
                )

            inputs = inputs * self.cond_weight(conditioning)
            inputs = inputs + self.cond_bias(conditioning)

        return inputs


class ConditionalInput(torch.nn.Module):
    """
    This module is used to condition any model inputs.
    If we don't have any conditions, this will be a normal pass.
    """

    def __init__(self, hidden_dim, condition_dim, condition_types=[]):
        check_support_condition_types(condition_types)
        super().__init__()
        self.support_types = ["add", "concat"]
        self.condition_types = [tp for tp in condition_types if tp in self.support_types]
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim

        if "add" in self.condition_types and condition_dim != hidden_dim:
            self.add_proj = torch.nn.Linear(condition_dim, hidden_dim)

        if "concat" in self.condition_types:
            self.concat_proj = torch.nn.Linear(hidden_dim + condition_dim, hidden_dim)

    def forward(self, inputs, conditioning=None):
        """
        Args:
            inputs (torch.tensor): B x T x C tensor.
            conditioning (torch.tensor): B x 1 x C conditioning embedding.
        """
        if len(self.condition_types) > 0:
            if conditioning is None:
                raise ValueError(
                    """You should add additional data types as conditions (e.g. speaker id or reference audio) 
                                 and define speaker_encoder in your config."""
                )

            if "add" in self.condition_types:
                if self.condition_dim != self.hidden_dim:
                    conditioning = self.add_proj(conditioning)
                inputs = inputs + conditioning

            if "concat" in self.condition_types:
                conditioning = conditioning.repeat(1, inputs.shape[1], 1)
                inputs = torch.cat([inputs, conditioning])
                inputs = self.concat_proj(inputs)

        return inputs

