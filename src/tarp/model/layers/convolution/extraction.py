from collections.abc import Sequence
from typing import final, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tarp.functional.activations.gated import swiglu
from tarp.model.layers.convolution.masked import DensityNormalizedConvolution1D


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
        self.kernel_sizes = tuple(kernel_sizes)

        self.experts = nn.ModuleList(
            [
                DensityNormalizedConvolution1D(
                    in_channels=model_dimension,
                    out_channels=model_dimension * 2,  # Dual projection
                    kernel_size=k,
                    padding=k // 2,
                    bias=bias,
                )
                for k in self.kernel_sizes
            ]
        )

        self.normalizations = nn.ModuleList(
            [nn.RMSNorm(model_dimension * 2) for _ in self.kernel_sizes]
        )

        self.router = nn.Linear(model_dimension, len(self.kernel_sizes))
        self.dropout = nn.Dropout(dropout)

    @override
    def forward(self, features: Tensor, attention_mask: Tensor) -> Tensor:
        attention_mask_expanded = attention_mask.unsqueeze(-1).to(features.dtype)

        # Compute the routing weights
        routing_weights = torch.softmax(self.router(features), dim=-1)

        signals = features.transpose(1, 2)  # [B, D, L]
        convolution_mask = attention_mask_expanded.transpose(1, 2)  # [B, 1, L]

        routed_output = torch.zeros_like(features)  # [B, L, D]

        for index, (expert, normalization) in enumerate(
            zip(self.experts, self.normalizations, strict=True)
        ):
            weight = routing_weights[..., index].unsqueeze(-1)  # [B, L, 1]

            transformed = expert(signals, convolution_mask).transpose(1, 2)
            normalized = normalization(transformed)  # [B, L, 2D]

            gate, value = torch.chunk(normalized, chunks=2, dim=-1)

            activated = self.dropout(swiglu(gate, value))  # [B, L, D]

            routed_output += weight * (activated * attention_mask_expanded)

        return routed_output + features  # Residual connection
