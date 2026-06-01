import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.original(x)
        lora_out = F.linear(self.dropout(x), self.lora_B @ self.lora_A)
        return result + lora_out * self.scaling

    def merge(self) -> nn.Linear:
        merged = nn.Linear(
            self.original.in_features,
            self.original.out_features,
            bias=self.original.bias is not None,
            device=self.original.weight.device,
            dtype=self.original.weight.dtype,
        )
        merged.weight.data = self.original.weight.data + (self.scaling * (self.lora_B @ self.lora_A))
        if self.original.bias is not None:
            merged.bias.data = self.original.bias.data.clone()
        return merged

    @property
    def in_features(self) -> int:
        return self.original.in_features

    @property
    def out_features(self) -> int:
        return self.original.out_features


class LoRAConv1d(nn.Module):
    def __init__(
        self,
        original_conv: nn.Conv1d,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original_conv
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        out_channels = original_conv.out_channels
        in_channels = original_conv.in_channels
        kernel_size = original_conv.kernel_size[0]
        stride = original_conv.stride[0]
        padding = original_conv.padding[0]
        dilation = original_conv.dilation[0]
        groups = original_conv.groups

        self.groups = groups

        self.lora_A = nn.Parameter(torch.empty(rank, in_channels // groups, kernel_size))
        self.lora_B = nn.Parameter(torch.zeros(out_channels, rank, 1))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.original(x)
        lora_input = self.dropout(x)
        lora_A_out = F.conv1d(
            lora_input,
            self.lora_A,
            None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        lora_out = F.conv1d(lora_A_out, self.lora_B, None, stride=1, padding=0)
        return result + lora_out * self.scaling

    def merge(self) -> nn.Conv1d:
        merged_weight = self.original.weight.data.clone()
        if self.groups == 1:
            delta = torch.einsum('or,rik->oik', self.lora_B.squeeze(-1), self.lora_A)
            merged_weight += self.scaling * delta
        else:
            for g in range(self.groups):
                cin_g = self.original.in_channels // self.groups
                cout_g = self.original.out_channels // self.groups
                w_g = self.original.weight.data[g * cout_g:(g + 1) * cout_g]
                rank_per_group = max(1, self.rank // self.groups) if self.rank >= self.groups else self.rank
                a_g = self.lora_A[g * rank_per_group:(g + 1) * rank_per_group]
                b_g = self.lora_B[g * cout_g:(g + 1) * cout_g]
                delta_g = torch.einsum('or,rik->oik', b_g.squeeze(-1), a_g)
                merged_weight[g * cout_g:(g + 1) * cout_g] += self.scaling * delta_g

        merged_conv = nn.Conv1d(
            self.original.in_channels,
            self.original.out_channels,
            self.original.kernel_size[0],
            stride=self.original.stride[0],
            padding=self.original.padding[0],
            dilation=self.original.dilation[0],
            groups=self.original.groups,
            bias=self.original.bias is not None,
            device=self.original.weight.device,
            dtype=self.original.weight.dtype,
        )
        merged_conv.weight.data = merged_weight
        if self.original.bias is not None:
            merged_conv.bias.data = self.original.bias.data.clone()
        return merged_conv


class LoRAConvTranspose1d(nn.Module):
    def __init__(
        self,
        original_conv: nn.ConvTranspose1d,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original_conv
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        out_channels = original_conv.out_channels
        in_channels = original_conv.in_channels
        kernel_size = original_conv.kernel_size[0]
        stride = original_conv.stride[0]
        padding = original_conv.padding[0]

        self.lora_A = nn.Parameter(torch.empty(rank, in_channels, 1))
        self.lora_B = nn.Parameter(torch.zeros(out_channels, rank, kernel_size))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.stride = stride
        self.padding = padding

        if self.original.weight.is_leaf:
            self.original.weight.requires_grad_(False)
        if self.original.bias is not None and self.original.bias.is_leaf:
            self.original.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.original(x)
        lora_input = self.dropout(x)
        lora_A_out = F.linear(lora_input.transpose(1, 2), self.lora_A.squeeze(2)).transpose(1, 2)
        lora_out = F.conv_transpose1d(
            lora_A_out,
            self.lora_B,
            None,
            stride=self.stride,
            padding=self.padding,
        )
        return result + lora_out * self.scaling

    def merge(self) -> nn.ConvTranspose1d:
        merged_weight = self.original.weight.data.clone()
        b = self.lora_B  # (out_channels, rank, kernel_size)
        a = self.lora_A  # (rank, in_channels, 1)
        k = self.original.kernel_size[0]
# ConvTranspose1d weight shape: (in_channels, out_channels, kernel_size)
        # delta needs to be (in_channels, out_channels, kernel_size)
        # b is (out_channels, rank, K), a is (rank, in_channels, 1) -> expanded to (rank, in_channels, K)
        # We want: delta[i,o,k] = sum_r a[r,i,k] * b[o,r,k]
        a_expanded = a.expand(-1, -1, k)  # (rank, in_channels, K)
        delta = torch.einsum('rik,ork->iok', a_expanded, b)  # (in_channels, out_channels, K)
        merged_weight += self.scaling * delta

        merged_conv = nn.ConvTranspose1d(
            self.original.in_channels,
            self.original.out_channels,
            self.original.kernel_size[0],
            stride=self.original.stride[0],
            padding=self.original.padding[0],
            bias=self.original.bias is not None,
            device=self.original.weight.device,
            dtype=self.original.weight.dtype,
        )
        merged_conv.weight.data = merged_weight
        if self.original.bias is not None:
            merged_conv.bias.data = self.original.bias.data.clone()
        return merged_conv