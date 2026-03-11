import math
from collections.abc import Callable
from typing import Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tarp.functional.fft.convolution import fft_cross_correlation_1d
from tarp.model.layers.convolutional import Autopad, PaddingMode


class FastConvolutionalLayer1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[int, Autopad] = Autopad.VALID,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: PaddingMode = PaddingMode.CONSTANT,
        padding_value: float = 0.0,
        device=None,
        dtype=None,
        use_weight: bool = True,
    ):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("`in_channels` must be divisible by `groups`")

        if out_channels % groups != 0:
            raise ValueError("`out_channels` must be divisible by `groups`")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.bias: Optional[nn.Parameter] = None
        self.weight: Optional[nn.Parameter] = None
        self.padding_value = padding_value

        if isinstance(padding, Autopad):
            match padding:
                case Autopad.SAME:
                    self.padding = (
                        kernel_size // 2 + (kernel_size - 2 * (kernel_size // 2)) - 1,
                        kernel_size // 2,
                    )
                case Autopad.VALID:
                    self.padding = (0, 0)
                case Autopad.CAUSAL:
                    self.padding = (kernel_size - 1, 0)
        else:
            self.padding = (padding, padding)

        if use_weight:
            self.weight = nn.Parameter(
                torch.empty(
                    (out_channels, in_channels // groups, kernel_size),
                )
            )

        if bias:
            self.bias = nn.Parameter(
                torch.empty((out_channels,), device=device, dtype=dtype)
            )

        self.padding_operation = self._get_padding_layer()

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            if self.weight is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            else:
                fan_in = (self.in_channels // self.groups) * self.kernel_size
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _get_padding_layer(self) -> Callable[[Tensor], Tensor]:
        mode = self.padding_mode.value  # Get the string
        if self.padding_mode == PaddingMode.CONSTANT:
            return lambda x: F.pad(x, self.padding, mode=mode, value=self.padding_value)
        else:
            return lambda x: F.pad(x, self.padding, mode=mode)

    def forward(
        self, input: Tensor, external_weight: Optional[Tensor] = None
    ) -> Tensor:
        """
        :param Tensor input: Input tensor of shape `(B, C_i, L)`
        :return: Output tensor of shape `(B, C_o, L_out)`
        """
        # Use F.pad to apply padding to the input tensor
        # Padding mode can be 'constant', 'reflect', 'replicate', or 'circular'
        # But for convolution, we typically use 'constant' padding with 'zeros'

        weight = external_weight if external_weight is not None else self.weight

        if weight is None:
            raise RuntimeError(
                "Weight must be provided either as a parameter or as an argument"
            )

        # Apply padding to the input tensor

        return fft_cross_correlation_1d(
            self.padding_operation(input),
            weight,
            bias=self.bias,
            stride=self.stride,
            padding=0,  # Padding is already applied to the input
            dilation=self.dilation,
            groups=self.groups,
        )
