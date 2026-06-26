from collections.abc import Sequence
from typing import final, override

import torch
from torch import Tensor, nn

from tarp.model.layers.convolution.masked import DensityNormalizedConvolution1D


@final
class ConcentricConvolutionExpertStack1D(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        kernel_sizes: Sequence[int] = (1, 3, 5, 7, 9),
        bias: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                DensityNormalizedConvolution1D(
                    in_channels=model_dimension,
                    out_channels=model_dimension,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=bias,
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.normalizations = nn.ModuleList(
            [nn.RMSNorm(model_dimension) for _ in kernel_sizes]
        )
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    @override
    def forward(self, features: Tensor, attention_mask: Tensor) -> Tensor:
        """
        :param Tensor features: [B, L, D]
        :param Tensor attention_mask: [B, L], 1 for valid positions, 0 for padding
        :return:
            experts: [B, L, K, D]
            features: [B, L, D]
        """
        signals = features.transpose(1, 2)  # [B, D, L]

        experts: list[Tensor] = []
        for convolution, normalization in zip(
            self.experts,
            self.normalizations,
            strict=True,
        ):
            update = convolution(signals, attention_mask.unsqueeze(1))  # [B, D, L]
            expert = self.dropout(
                self.activation(normalization(update.transpose(1, 2)))
            )  # [B, L, D]
            expert = expert * attention_mask.unsqueeze(-1).to(dtype=expert.dtype)
            experts.append(expert)
        stacked = torch.stack(experts, dim=2)  # [B, L, K, D]
        return stacked


@final
class ReceptiveFieldGating(nn.Module):
    def __init__(self, model_dimension: int, number_of_experts: int):
        super().__init__()
        self.gate = nn.Linear(model_dimension, number_of_experts)

    @override
    def forward(self, features: Tensor, experts: Tensor) -> Tensor:
        """
        :param Tensor features: [B, L, D] - the original input features, which are used to compute the gating weights for routing
        :param Tensor experts: [B, L, K, D] - the output of the expert stack, where K is the number of experts (i.e., different convolutional kernel sizes)
        :return Tensor: [B, L, D] - the output of the router after combining the experts according to the gating weights
        """
        weights = self.gate(features).softmax(dim=-1)  # [B, L, K]
        # [B, L, K] x [B, L, K, D] -> [B, L, D]
        routed = torch.einsum("blk,blkd->bld", weights, experts)
        return routed


@final
class AdaptiveReceptiveField1D(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        kernel_sizes: Sequence[int] = (1, 3, 5, 7, 9),
        bias: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.expert_stack = ConcentricConvolutionExpertStack1D(
            model_dimension=model_dimension,
            kernel_sizes=kernel_sizes,
            bias=bias,
            dropout=dropout,
        )
        self.router = ReceptiveFieldGating(
            model_dimension=model_dimension,
            number_of_experts=len(kernel_sizes),
        )

    @override
    def forward(self, features: Tensor, attention_mask: Tensor) -> Tensor:
        experts = self.expert_stack(features, attention_mask)  # [B, L, K, D]
        routed = self.router(features, experts)  # [B, L, D]
        return routed
