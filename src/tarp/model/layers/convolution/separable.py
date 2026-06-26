from typing import final, override

from torch import Tensor, nn


@final
class DepthwiseSeparableConvolution1D(nn.Module):
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
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, groups=groups, bias=bias
        )

    @override
    def forward(self, features: Tensor) -> Tensor:
        """
        :param Tensor features: Shape [B, C_in, L_in]
        :return Tensor: Tensor of shape [B, C_out, L_out]
        """
        depthwise_output = self.depthwise(features)
        pointwise_output = self.pointwise(depthwise_output)
        return pointwise_output
