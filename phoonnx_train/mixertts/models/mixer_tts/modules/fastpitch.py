# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
# https://github.com/NVIDIA/NeMo/blob/7256db10771aa1d213d9b49640667efaa14f89c9/nemo/collections/tts/modules/fastpitch.py

import torch
import torch.nn as nn

from .submodules import ConditionalInput, ConditionalLayerNorm
# from nemo.core.classes import NeuralModule, adapter_mixins, typecheck
# from nemo.core.neural_types.elements import (
#     EncodedRepresentation,
#     Index,
#     LengthsType,
#     LogprobsType,
#     MelSpectrogramType,
#     ProbsType,
#     RegressionValuesType,
#     TokenDurationType,
#     TokenIndex,
#     TokenLogDurationType,
# )
# from nemo.core.neural_types.neural_type import NeuralType


def average_features(pitch, durs):
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
    pitch_nonzero_cums = torch.nn.functional.pad(torch.cumsum(pitch != 0.0, dim=2), (1, 0))
    pitch_cums = torch.nn.functional.pad(torch.cumsum(pitch, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = pitch.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    pitch_sums = (torch.gather(pitch_cums, 2, dce) - torch.gather(pitch_cums, 2, dcs)).float()
    pitch_nelems = (torch.gather(pitch_nonzero_cums, 2, dce) - torch.gather(pitch_nonzero_cums, 2, dcs)).float()

    pitch_avg = torch.where(pitch_nelems == 0.0, pitch_nelems, pitch_sums / pitch_nelems)
    return pitch_avg


def log_to_duration(log_dur, min_dur, max_dur, mask):
    dur = torch.clamp(torch.exp(log_dur) - 1.0, min_dur, max_dur)
    dur *= mask.squeeze(2)
    return dur


class ConvReLUNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dropout=0.0, condition_dim=384, condition_types=[]):
        super(ConvReLUNorm, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size // 2))
        self.norm = ConditionalLayerNorm(out_channels, condition_dim=condition_dim, condition_types=condition_types)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, signal, conditioning=None):
        out = torch.nn.functional.relu(self.conv(signal))
        out = self.norm(out.transpose(1, 2), conditioning).transpose(1, 2)
        out = self.dropout(out)

        # if self.is_adapter_available():
        #     out = self.forward_enabled_adapters(out.transpose(1, 2)).transpose(1, 2)

        return out


class TemporalPredictor(nn.Module):
    """Predicts a single float per each temporal location"""

    def __init__(self, input_size, filter_size, kernel_size, dropout, n_layers=2, condition_types=[]):
        super(TemporalPredictor, self).__init__()
        self.cond_input = ConditionalInput(input_size, input_size, condition_types)
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                ConvReLUNorm(
                    input_size if i == 0 else filter_size,
                    filter_size,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    condition_dim=input_size,
                    condition_types=condition_types,
                )
            )
        self.fc = torch.nn.Linear(filter_size, 1, bias=True)

        # Use for adapter input dimension
        self.filter_size = filter_size

    # @property
    # def input_types(self):
    #     return {
    #         "enc": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
    #         "enc_mask": NeuralType(('B', 'T', 1), TokenDurationType()),
    #         "conditioning": NeuralType(('B', 'T', 'D'), EncodedRepresentation(), optional=True),
    #     }

    # @property
    # def output_types(self):
    #     return {
    #         "out": NeuralType(('B', 'T'), EncodedRepresentation()),
    #     }

    def forward(self, enc, enc_mask, conditioning=None):
        enc = self.cond_input(enc, conditioning)
        out = enc * enc_mask
        out = out.transpose(1, 2)

        for layer in self.layers:
            out = layer(out, conditioning=conditioning)

        out = out.transpose(1, 2)
        out = self.fc(out) * enc_mask
        return out.squeeze(-1)


