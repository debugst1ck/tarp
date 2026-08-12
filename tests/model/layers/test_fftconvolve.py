import pytest
import torch
import torch.nn.functional as F

from tarp.functional.fft.convolution import fft_cross_correlation_1d

# Test case parameters: (name, B, C_i, C_o, L, K, stride, padding, dilation, groups, bias)
TEST_CASES = [
    # Basic tests - no special parameters
    ("tiny_basic", 1, 1, 1, 8, 3, 1, 0, 1, 1, False),
    ("small_basic", 2, 3, 5, 16, 3, 1, 0, 1, 1, False),
    ("medium_basic", 4, 8, 16, 32, 5, 1, 0, 1, 1, False),
    ("large_basic", 2, 16, 32, 128, 7, 1, 0, 1, 1, False),
    # Padding tests
    ("tiny_padded", 1, 1, 1, 8, 3, 1, 1, 1, 1, False),
    ("small_padded", 2, 3, 5, 16, 3, 1, 1, 1, 1, False),
    ("medium_padded", 4, 8, 16, 32, 5, 1, 2, 1, 1, False),
    ("large_padded", 2, 16, 32, 128, 7, 1, 3, 1, 1, False),
    ("zero_padding", 2, 4, 8, 20, 5, 1, 0, 1, 1, False),
    ("large_padding", 1, 4, 4, 20, 3, 1, 10, 1, 1, False),
    # Stride tests
    ("tiny_stride2", 1, 1, 1, 8, 3, 2, 0, 1, 1, False),
    ("small_stride2", 2, 3, 5, 16, 3, 2, 0, 1, 1, False),
    ("medium_stride2", 4, 8, 16, 32, 5, 2, 0, 1, 1, False),
    ("stride3", 3, 6, 12, 48, 3, 3, 1, 1, 1, False),
    ("stride4", 1, 2, 2, 20, 5, 4, 2, 1, 1, False),
    ("stride5", 2, 3, 5, 50, 3, 5, 1, 1, 1, False),
    ("large_stride", 1, 2, 4, 100, 7, 10, 3, 1, 1, False),
    # Dilation tests
    ("tiny_dilation2", 1, 1, 1, 8, 3, 1, 0, 2, 1, False),
    ("small_dilation2", 2, 3, 5, 16, 3, 1, 0, 2, 1, False),
    ("medium_dilation2", 4, 8, 16, 32, 5, 1, 0, 2, 1, False),
    ("dilation3", 2, 4, 8, 40, 3, 1, 2, 3, 1, False),
    ("dilation4", 1, 2, 4, 50, 5, 1, 5, 4, 1, False),
    # Groups tests
    ("groups2", 2, 4, 8, 32, 3, 1, 0, 1, 2, False),
    ("groups4", 2, 8, 16, 32, 3, 1, 0, 1, 4, False),
    ("groups8", 1, 16, 32, 64, 5, 1, 2, 1, 8, False),
    ("depthwise_conv", 2, 8, 8, 32, 3, 1, 0, 1, 8, False),
    ("depthwise_large", 1, 16, 16, 64, 7, 1, 3, 1, 16, False),
    # Bias tests
    ("tiny_bias", 1, 1, 1, 8, 3, 1, 0, 1, 1, True),
    ("small_bias", 2, 3, 5, 16, 3, 1, 0, 1, 1, True),
    ("medium_bias", 4, 8, 16, 32, 5, 1, 0, 1, 1, True),
    ("bias_with_stride", 2, 4, 8, 40, 3, 2, 1, 1, 1, True),
    ("bias_with_groups", 2, 4, 8, 32, 3, 1, 0, 1, 2, True),
    # Combined parameters
    ("stride_padding", 2, 4, 8, 40, 5, 2, 2, 1, 1, False),
    ("stride_dilation", 1, 3, 6, 50, 3, 2, 0, 2, 1, False),
    ("stride_groups", 2, 4, 8, 40, 3, 2, 1, 1, 2, False),
    ("padding_dilation", 2, 4, 8, 40, 5, 1, 3, 2, 1, False),
    ("padding_groups", 2, 4, 8, 32, 3, 1, 2, 1, 2, False),
    ("dilation_groups", 2, 4, 8, 40, 3, 1, 0, 2, 2, False),
    # Complex combinations
    ("complex1", 2, 4, 8, 64, 5, 2, 2, 1, 2, True),
    ("complex2", 3, 6, 12, 48, 3, 3, 1, 2, 3, False),
    ("complex3", 1, 8, 16, 100, 7, 1, 3, 2, 1, True),
    ("complex4", 2, 8, 16, 80, 5, 2, 1, 2, 4, True),
    ("complex5", 1, 12, 24, 120, 7, 3, 2, 1, 6, False),
    # Edge cases
    ("kernel_size1", 2, 4, 4, 16, 1, 1, 0, 1, 1, False),
    ("signal_length1", 1, 2, 2, 1, 1, 1, 0, 1, 1, False),
    ("signal_length1_padded", 1, 2, 2, 1, 1, 1, 2, 1, 1, False),
    ("tiny_signal_large_kernel", 1, 2, 4, 5, 3, 1, 0, 1, 1, False),
    ("exact_match", 1, 2, 4, 5, 5, 1, 0, 1, 1, False),
    ("larger_kernel", 1, 3, 6, 20, 11, 1, 0, 1, 1, False),
    # Stress tests
    ("batch_size1", 1, 8, 16, 64, 5, 2, 1, 1, 2, False),
    ("batch_size8", 8, 4, 8, 32, 3, 1, 0, 1, 1, False),
    ("many_channels", 2, 32, 64, 50, 5, 1, 2, 1, 1, False),
    ("long_signal", 1, 4, 8, 500, 7, 1, 3, 1, 1, False),
    ("long_signal_stride", 1, 4, 8, 500, 7, 5, 3, 1, 1, False),
    # Additional stride variations
    ("odd_stride_odd_kernel", 2, 4, 8, 33, 5, 3, 1, 1, 1, False),
    ("even_stride_odd_kernel", 2, 4, 8, 32, 5, 4, 2, 1, 1, False),
    ("odd_stride_even_kernel", 2, 4, 8, 33, 4, 3, 1, 1, 1, False),
    ("even_stride_even_kernel", 2, 4, 8, 32, 4, 2, 1, 1, 1, False),
    # Additional dilation variations
    ("dilation_with_padding", 2, 4, 8, 40, 3, 1, 4, 2, 1, False),
    ("dilation_with_stride", 2, 4, 8, 50, 3, 3, 2, 2, 1, False),
    ("large_dilation", 1, 3, 6, 60, 3, 1, 5, 5, 1, False),
    # Additional groups variations
    ("groups_with_stride", 2, 6, 12, 48, 3, 2, 1, 1, 3, False),
    ("groups_with_dilation", 2, 6, 12, 48, 3, 1, 2, 2, 3, False),
    ("groups_with_padding", 2, 6, 12, 48, 5, 1, 3, 1, 3, False),
    # Comprehensive combinations
    ("all_params1", 2, 6, 12, 60, 5, 2, 2, 2, 3, True),
    ("all_params2", 1, 8, 16, 80, 7, 3, 3, 2, 4, True),
    ("all_params3", 3, 12, 24, 100, 5, 2, 1, 3, 6, False),
]


@pytest.mark.parametrize(
    "name,B,C_i,C_o,L,K,stride,padding,dilation,groups,use_bias",
    TEST_CASES,
    ids=[case[0] for case in TEST_CASES],
)
def test_fft_conv1d(
    name, B, C_i, C_o, L, K, stride, padding, dilation, groups, use_bias
):
    """
    Test FFTConvolve1D against PyTorch's F.conv1d for correctness.

    Tests shape matching and numerical accuracy with reasonable floating-point tolerances.
    """
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create input and filter tensors
    input_tensor = torch.randn(B, C_i, L)
    filter_tensor = torch.randn(C_o, C_i // groups, K)
    bias_tensor = torch.randn(C_o) if use_bias else None

    # Compute using FFT implementation
    fft_output = fft_cross_correlation_1d(
        input_tensor,
        filter_tensor,
        bias_tensor,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    # Compute using PyTorch reference
    torch_output = F.conv1d(
        input_tensor,
        filter_tensor,
        bias_tensor,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    # Check shape
    assert fft_output.shape == torch_output.shape, (
        f"Shape mismatch: FFT={fft_output.shape}, PyTorch={torch_output.shape}"
    )

    # Check numerical accuracy
    max_diff = torch.max(torch.abs(fft_output - torch_output)).item()
    rel_error = (
        torch.norm(fft_output - torch_output) / torch.norm(torch_output)
    ).item()

    # Use reasonable tolerances for floating-point comparison
    assert max_diff < 1e-4, (
        f"Max difference too large: {max_diff:.2e} "
        f"(params: B={B}, C_i={C_i}, C_o={C_o}, L={L}, K={K}, "
        f"stride={stride}, padding={padding}, dilation={dilation}, groups={groups}, bias={use_bias})"
    )

    assert rel_error < 1e-4, (
        f"Relative error too large: {rel_error:.2e} "
        f"(params: B={B}, C_i={C_i}, C_o={C_o}, L={L}, K={K}, "
        f"stride={stride}, padding={padding}, dilation={dilation}, groups={groups}, bias={use_bias})"
    )


# Additional test for specific debugging
def test_fft_conv1d_debug_stride():
    """Debug test to verify stride behavior with simple example"""
    torch.manual_seed(42)

    B, C_i, C_o, L, K = 1, 1, 1, 10, 3
    stride = 2

    input_tensor = torch.randn(B, C_i, L)
    filter_tensor = torch.randn(C_o, C_i, K)

    fft_output = fft_cross_correlation_1d(input_tensor, filter_tensor, stride=stride)
    torch_output = F.conv1d(input_tensor, filter_tensor, stride=stride)

    assert fft_output.shape == torch_output.shape
    assert torch.allclose(fft_output, torch_output, atol=1e-4, rtol=1e-4)


def test_fft_conv1d_zero_input():
    """Test with zero input"""
    input_tensor = torch.zeros(1, 2, 10)
    filter_tensor = torch.randn(4, 2, 3)

    fft_output = fft_cross_correlation_1d(input_tensor, filter_tensor)
    torch_output = F.conv1d(input_tensor, filter_tensor)

    assert torch.allclose(fft_output, torch_output, atol=1e-6)


def test_fft_conv1d_zero_filter():
    """Test with zero filter"""
    input_tensor = torch.randn(1, 2, 10)
    filter_tensor = torch.zeros(4, 2, 3)

    fft_output = fft_cross_correlation_1d(input_tensor, filter_tensor)
    torch_output = F.conv1d(input_tensor, filter_tensor)

    assert torch.allclose(fft_output, torch_output, atol=1e-6)


def test_fft_conv1d_ones():
    """Test with all ones"""
    input_tensor = torch.ones(2, 3, 15)
    filter_tensor = torch.ones(5, 3, 3)

    fft_output = fft_cross_correlation_1d(input_tensor, filter_tensor)
    torch_output = F.conv1d(input_tensor, filter_tensor)

    assert torch.allclose(fft_output, torch_output, atol=1e-4, rtol=1e-4)
