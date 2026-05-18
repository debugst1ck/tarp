from typing import override

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DensityNormalizedConvolution1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.feature_convolution = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    @override
    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        """
        :param Tensor features: Shape [B, C_in, L_in]
        :param Tensor mask: Shape [B, C_in, L_in] or [B, 1, L_in] where values are 0 for masked positions and 1 for valid positions
        :return Tensor: Tensor of shape [B, C_out, L_out]
        """
        mask = mask.to(dtype=features.dtype)  # [B, C_in, L_in]
        masked_features = features * mask  # [B, C_in, L_in]
        spatial_mask = mask.sum(dim=1, keepdim=True)  # [B, 1, L_in]
        ones_weight = torch.ones(
            (1, 1, self.kernel_size),
            dtype=features.dtype,
            device=features.device,
        )  # [1, 1, kernel_size]
        feature_density = F.conv1d(
            spatial_mask,
            weight=ones_weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        ).clamp_min(1.0)  # [B, 1, L_out]
        output = self.feature_convolution(masked_features)  # [B, C_out, L_out]
        normalization_factor = feature_density / (
            self.kernel_size * mask.size(1)
        )  # [B, 1, L_out]
        return output / (normalization_factor)  # [B, C_out, L_out]
