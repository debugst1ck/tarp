from typing import Optional

import scipy.fft
import torch.fft
import torch.nn.functional as F
from torch import Tensor


def next_power_of_two(n: int) -> int:
    """Returns the next power of 2 greater than or equal to n."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    return 1 << (n - 1).bit_length()


def fft_cross_correlation_nd(
    input: Tensor,
    filter: Tensor,
    bias: Optional[Tensor] = None,
    stride: tuple[int, ...] = (1,),
    padding: tuple[int, ...] = (0,),
    dilation: tuple[int, ...] = (1,),
    groups: int = 1,
) -> Tensor:
    """
    An N-dimensional cross-correlation using FFT.
    The output shape in each dimension d is computed as:
    L_out[d] = floor((L[d] + 2*padding[d] - dilation[d]*(K[d]-1) - 1) / stride[d]) + 1

    :param Tensor input: Input tensor of shape `(B, C_i, L_1, L_2, ..., L_N)`
    :param Tensor filter: Filter tensor of shape `(C_o, C_i // G, K_1, K_2, ..., K_N)`, where G is the number of groups
    :param Optional[Tensor] bias: Bias tensor of shape `(C_o,)` or None
    :param tuple[int, ...] stride: Step sizes in which convolution window moves in each dimension
    :param tuple[int, ...] padding: Amount of zero-padding added to both sides of input in each dimension (use `K[d]//2` for 'same' padding)
    :param tuple[int, ...] dilation: Spacing between kernel elements in each dimension, expands receptive field without increasing kernel size
    :param int groups: Number of blocked connections from input channels to output channels. `C_i` and `C_o` must be divisible by `G`
    :return Tensor: Convolved tensor of shape `(B, C_o, L_out_1, L_out_2, ..., L_out_N)`
    """

    # Get the number of spatial dimensions
    dimensions = input.dim() - 2  # Exclude batch and channel dimensions
    batch_size, in_channel = input.shape[:2]
    out_channels, filter_in_channel, *kernel_sizes = filter.shape
    signal_sizes = input.shape[2:]

    # filter_in_channels already equals input_channels // groups due to convND requirements
    group_in_channel = filter_in_channel
    group_out_channel = out_channels // groups

    input_groups = input.reshape(batch_size, groups, group_in_channel, *signal_sizes)
    filter_groups = filter.reshape(
        groups, group_out_channel, group_in_channel, *kernel_sizes
    )

    # Check that the dimensions of stride, padding, and dilation match the number of spatial dimensions
    if not (len(stride) == len(padding) == len(dilation) == dimensions):
        raise ValueError(
            "Length of stride, padding, and dilation must match the number of spatial dimensions"
        )

    # Flip the kernel for convolution across all spatial dimensions
    # Technically, cross-correlation does not require flipping
    # But multiplying in frequency domain is circular convolution, which gives true convolution
    # So we flip the kernel here to achieve cross-correlation via FFT
    # But wait, we can take conj of FFT of kernel instead of flipping in time domain
    # This saves a computation step
    # filter_groups = filter_groups.flip(
    #     tuple(range(-dimensions, 0))
    # )  # (G, C_o/G, C_i/G, K_1, K_2, ..., K_N)

    # Dilate the kernel if needed
    if any(d > 1 for d in dilation):
        dilated_kernel_sizes = tuple(
            k + (k - 1) * (d - 1) for k, d in zip(kernel_sizes, dilation)
        )
        new_kernel = torch.zeros(
            (groups, group_out_channel, group_in_channel, *dilated_kernel_sizes),
            device=filter_groups.device,
            dtype=filter_groups.dtype,
        )
        # Create slices for inserting the original kernel into the dilated kernel
        slices = [slice(0, d * k, d) for k, d in zip(kernel_sizes, dilation)]

        new_kernel[..., *slices] = filter_groups
        filter_groups = new_kernel  # (G, C_o/G, C_i/G, K_d_1, K_d_2, ..., K_d_N)
        kernel_sizes = dilated_kernel_sizes  # Update kernel sizes

    # Pad the input in all spatial dimensions
    padding_tuple = [x for p in reversed(padding) for x in (p, p)]
    padded_input = F.pad(input_groups, padding_tuple)

    # Determine the size for FFT per dimension, use next fast length for normal efficiency
    # But cuFFT only supports powers of 2, so we use that for GPU tensors
    fft_sizes: tuple[int, ...] = tuple(
        [
            scipy.fft.next_fast_len(
                padded_input.size(dimension + 3) + kernel_sizes[dimension] - 1
            )  # type: ignore
            if not input.is_cuda
            else next_power_of_two(
                padded_input.size(dimension + 3) + kernel_sizes[dimension] - 1
            )
            for dimension in range(dimensions)
        ]
    )

    # Perform FFT on input and filter
    # Define dimensions over which to perform FFT
    fft_dimensions = tuple(range(-dimensions, 0))

    input_fft: Tensor = torch.fft.rfftn(
        padded_input, s=fft_sizes, dim=fft_dimensions
    )  # (B, G, C_i/G, F_1, F_2, ..., F_N)

    filter_fft: Tensor = torch.fft.rfftn(
        filter_groups, s=fft_sizes, dim=fft_dimensions
    )  # (G, C_o/G, C_i/G, F_1, F_2, ..., F_N)

    # Multiply in frequency domain and sum over input channels within each group
    # Using broadcasting to align dimensions for multiplication
    # Flip is handled by taking conjugate of filter_fft, to achieve cross-correlation
    # input_fft: (B, G, C_i/G, F_1, F_2, ..., F_N) -> (B, G, C_i/G, 1, F_1, F_2, ..., F_N)
    output_fft = (input_fft.unsqueeze(2) * filter_fft.conj().unsqueeze(0)).sum(dim=3)

    # Inverse rfftn to get time-domain result
    output: Tensor = torch.fft.irfftn(
        output_fft, s=fft_sizes, dim=fft_dimensions
    ).real  # (B, G, C_o/G, fft_size_1, fft_size_2, ..., fft_size_N)

    # Crop to the valid length in each dimension
    # Calculate start and end indices for each dimension
    starts = [0 for k in kernel_sizes]
    output_lengths = [
        (padded_input.size(d + 3) - k) // s + 1
        for d, (k, s) in enumerate(zip(kernel_sizes, stride))
    ]
    ends = [
        start + length * s for start, length, s in zip(starts, output_lengths, stride)
    ]

    # Create slices for cropping and striding
    slices = [slice(start, end, s) for start, end, s in zip(starts, ends, stride)]
    output = output[(..., *slices)]  # (B, G, C_o/G, L_out_1, L_out_2, ..., L_out_N)

    # Reshape to merge groups back into output channels
    output = output.reshape(
        batch_size, groups * group_out_channel, *output_lengths
    )  # (B, C_o, L_out_1, L_out_2, ..., L_out_N)

    # Add bias if provided
    if bias is not None:
        output += bias.reshape(1, -1, *[1] * dimensions)

    return output.to(input.dtype)


def fft_cross_correlation_1d(
    input: Tensor,
    filter: Tensor,
    bias: Optional[Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> Tensor:
    """
    A 1D convolution using FFT.

    The output length L_out is computed as:
    L_out = floor((L + 2*padding - dilation*(K-1) - 1) / stride) + 1

    :param Tensor input: Input tensor of shape `(B, C_i, L)`
    :param Tensor filter: Filter tensor of shape `(C_o, C_i // G, K)`, where G is the number of groups
    :param Optional[Tensor] bias: Bias tensor of shape `(C_o,)` or None
    :param int stride: Step size in which convolution window moves
    :param int padding: Amount of zero-padding added to both sides of input (use `K//2` for 'same' padding)
    :param int dilation: Spacing between kernel elements, expands receptive field without increasing kernel size
    :param int groups: Number of blocked connections from input channels to output channels. `C_i` and `C_o` must be divisible by `G`
    :return Tensor: Convolved tensor of shape `(B, C_o, L_out)`
    """
    return fft_cross_correlation_nd(
        input,
        filter,
        bias,
        stride=(stride,),
        padding=(padding,),
        dilation=(dilation,),
        groups=groups,
    )


def fft_cross_correlation_2d(
    input: Tensor,
    filter: Tensor,
    bias: Optional[Tensor] = None,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    groups: int = 1,
) -> Tensor:
    """
    A 2D convolution using FFT.

    The output shape in each dimension d is computed as:
    L_out[d] = floor((L[d] + 2*padding[d] - dilation[d]*(K[d]-1) - 1) / stride[d]) + 1

    :param Tensor input: Input tensor of shape `(B, C_i, H, W)`
    :param Tensor filter: Filter tensor of shape `(C_o, C_i // G, K_H, K_W)`, where G is the number of groups
    :param Optional[Tensor] bias: Bias tensor of shape `(C_o,)` or None
    :param tuple[int, int] stride: Step sizes in which convolution window moves in height and width dimensions
    :param tuple[int, int] padding: Amount of zero-padding added to both sides of input in height and width dimensions (use `K[d]//2` for 'same' padding)
    :param tuple[int, int] dilation: Spacing between kernel elements in height and width dimensions, expands receptive field without increasing kernel size
    :param int groups: Number of blocked connections from input channels to output channels. `C_i` and `C_o` must be divisible by `G`
    :return Tensor: Convolved tensor of shape `(B, C_o, H_out, W_out)`
    """
    return fft_cross_correlation_nd(
        input,
        filter,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
