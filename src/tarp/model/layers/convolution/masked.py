from typing import final, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tarp.model.layers.convolution.separable import DepthwiseSeparableConvolution1D


@final
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
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.feature_convolution = DepthwiseSeparableConvolution1D(
            in_channels,
            out_channels,
            kernel_size,
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
        :param Tensor mask: Shape [B, 1, L_in] where values are 0 for masked positions and 1 for valid positions
        :return Tensor: Tensor of shape [B, C_out, L_out]
        """
        mask = mask.to(dtype=features.dtype)
        masked_features = features * mask

        density = F.conv1d(
            mask,
            weight=torch.ones(
                1,
                1,
                self.kernel_size,
                dtype=features.dtype,
                device=features.device,
            ),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        ).clamp_min(1.0)

        valid_fraction = density / self.kernel_size

        output = self.feature_convolution(masked_features)
        output = output / valid_fraction
        return output
