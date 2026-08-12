from collections.abc import Sequence
from enum import Enum


class Autopad(Enum):
    SAME = "same"
    VALID = "valid"
    CAUSAL = "causal"
    FULL = "full"


class PaddingMode(Enum):
    CONSTANT = "constant"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"


def autopad_nd(
    mode: Autopad,
    kernel_sizes: Sequence[int],
    strides: Sequence[int],
    dilations: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    """
    Compute padding for 'same', 'valid', 'causal', and 'full' modes in N dimensions.

    :param Autopad mode: Padding mode to compute
    :param Sequence[int, ...] kernel_sizes: Kernel sizes for each dimension
    :param Sequence[int, ...] strides: Strides for each dimension
    :param Sequence[int, ...] dilations: Dilations for each dimension
    :return tuple[tuple[int, int], ...]: Padding for each dimension as a tuple of (pad_left, pad_right)
    """
    if len(kernel_sizes) != len(strides) or len(kernel_sizes) != len(dilations):
        raise ValueError(
            "`kernel_sizes`, `strides`, and `dilations` must have the same length"
        )
    padding: list[tuple[int, int]] = []
    for k, s, d in zip(kernel_sizes, strides, dilations):
        effective_kernel_size = (k - 1) * d + 1
        match mode:
            case Autopad.SAME:
                if s > 1:
                    raise ValueError("Autopad.SAME is not compatible with stride > 1")
                total_padding = effective_kernel_size - 1
                pad_left = total_padding // 2
                pad_right = total_padding - pad_left
                padding.append((pad_left, pad_right))
            case Autopad.VALID:
                padding.append((0, 0))
            case Autopad.CAUSAL:
                padding.append((effective_kernel_size - 1, 0))
            case Autopad.FULL:
                padding.append((effective_kernel_size - 1, effective_kernel_size - 1))
    return tuple(padding)
