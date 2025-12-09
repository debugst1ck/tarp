import pytest
import torch
import torch.nn.functional as F

from tarp.functional.fft.convolution import fft_cross_correlation_nd

# ------------------------------------------------------------
# Parametric test cases for FFTConvolveND
# Format:
# (name, dims, B, C_i, C_o, sizes, kernel_sizes, stride, padding, dilation, groups, bias)
# ------------------------------------------------------------

TEST_CASES_ND = [
    # 2D basic
    ("2d_basic_small", 2, 1, 3, 4, (16, 16), (3, 3), (1, 1), (0, 0), (1, 1), 1, False),
    ("2d_basic_med", 2, 2, 6, 8, (32, 20), (5, 3), (1, 1), (0, 0), (1, 1), 1, False),
    ("2d_basic_large", 2, 2, 8, 16, (64, 64), (7, 5), (1, 1), (0, 0), (1, 1), 1, False),
    # 2D padding
    ("2d_padding1", 2, 1, 3, 3, (20, 20), (3, 3), (1, 1), (1, 1), (1, 1), 1, False),
    ("2d_padding2", 2, 2, 4, 4, (32, 40), (5, 3), (1, 1), (2, 1), (1, 1), 1, False),
    # 2D stride
    ("2d_stride2", 2, 1, 3, 4, (20, 20), (3, 3), (2, 2), (0, 0), (1, 1), 1, False),
    ("2d_stride3", 2, 1, 3, 4, (30, 30), (5, 5), (3, 3), (2, 2), (1, 1), 1, False),
    # 2D dilation
    ("2d_dilation2", 2, 1, 3, 4, (32, 32), (3, 3), (1, 1), (0, 0), (2, 2), 1, False),
    ("2d_dilation3", 2, 2, 4, 6, (24, 24), (3, 3), (1, 1), (1, 1), (3, 2), 1, False),
    # 2D groups
    ("2d_groups2", 2, 2, 4, 8, (32, 32), (3, 3), (1, 1), (0, 0), (1, 1), 2, False),
    ("2d_groups4", 2, 1, 8, 16, (16, 16), (5, 5), (1, 1), (2, 2), (1, 1), 4, False),
    # 2D bias
    ("2d_bias", 2, 1, 3, 6, (16, 16), (3, 3), (1, 1), (0, 0), (1, 1), 1, True),
    # 2D combined
    ("2d_combo", 2, 2, 6, 12, (40, 40), (5, 3), (2, 1), (2, 1), (2, 1), 3, True),
    # --------------------------
    # 3D Cases
    # --------------------------
    (
        "3d_basic",
        3,
        1,
        2,
        4,
        (16, 10, 8),
        (3, 3, 3),
        (1, 1, 1),
        (0, 0, 0),
        (1, 1, 1),
        1,
        False,
    ),
    (
        "3d_padding",
        3,
        1,
        3,
        6,
        (20, 12, 10),
        (3, 3, 3),
        (1, 1, 1),
        (1, 2, 1),
        (1, 1, 1),
        1,
        False,
    ),
    (
        "3d_stride",
        3,
        1,
        2,
        4,
        (24, 16, 12),
        (3, 3, 3),
        (2, 2, 2),
        (0, 0, 0),
        (1, 1, 1),
        1,
        False,
    ),
    (
        "3d_dilation",
        3,
        1,
        4,
        8,
        (24, 24, 24),
        (3, 3, 3),
        (1, 1, 1),
        (0, 0, 0),
        (2, 2, 2),
        1,
        False,
    ),
    (
        "3d_groups",
        3,
        1,
        8,
        8,
        (20, 20, 20),
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        8,
        False,
    ),
    (
        "3d_bias",
        3,
        1,
        3,
        6,
        (10, 10, 10),
        (3, 3, 3),
        (1, 1, 1),
        (0, 0, 0),
        (1, 1, 1),
        1,
        True,
    ),
    # 3D combined
    (
        "3d_combo",
        3,
        2,
        6,
        12,
        (20, 16, 12),
        (5, 3, 3),
        (2, 1, 2),
        (2, 1, 1),
        (2, 1, 1),
        3,
        True,
    ),
]


@pytest.mark.parametrize(
    "name,dims,B,C_i,C_o,sizes,kernel_sizes,stride,padding,dilation,groups,use_bias",
    TEST_CASES_ND,
    ids=[case[0] for case in TEST_CASES_ND],
)
def test_fft_conv_nd(
    name,
    dims,
    B,
    C_i,
    C_o,
    sizes,
    kernel_sizes,
    stride,
    padding,
    dilation,
    groups,
    use_bias,
):
    torch.manual_seed(1234)

    # Create random input and filter
    input_tensor = torch.randn(B, C_i, *sizes)
    filter_tensor = torch.randn(C_o, C_i // groups, *kernel_sizes)
    bias_tensor = torch.randn(C_o) if use_bias else None

    # FFT implementation
    fft_output = fft_cross_correlation_nd(
        input_tensor,
        filter_tensor,
        bias=bias_tensor,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    # PyTorch reference
    if dims == 2:
        torch_output = F.conv2d(
            input_tensor,
            filter_tensor,
            bias=bias_tensor,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    elif dims == 3:
        torch_output = F.conv3d(
            input_tensor,
            filter_tensor,
            bias=bias_tensor,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    else:
        raise ValueError("Unsupported dimensionality for this test suite")

    # Shape check
    assert fft_output.shape == torch_output.shape, (
        f"Shape mismatch in {name}: FFT={fft_output.shape}, Torch={torch_output.shape}"
    )

    # Numerical accuracy
    max_diff = (fft_output - torch_output).abs().max().item()
    rel_error = (
        torch.norm(fft_output - torch_output) / torch.norm(torch_output)
    ).item()

    assert max_diff < 1e-4, f"{name}: max diff too large: {max_diff}"
    assert rel_error < 1e-4, f"{name}: relative error too large: {rel_error}"


# ------------------------------------------------------------
# Additional special-case tests
# ------------------------------------------------------------


def test_fft_conv_nd_zero_input():
    x = torch.zeros(1, 3, 16, 16)
    w = torch.randn(6, 3, 3, 3)

    stride = (1, 1)
    padding = (0, 0)
    dilation = (1, 1)

    assert torch.allclose(
        fft_cross_correlation_nd(
            x,
            w,
            stride=stride,
            padding=padding,
            dilation=dilation,
        ),
        F.conv2d(x, w, stride=stride, padding=padding, dilation=dilation),
        atol=1e-6,
    )


def test_fft_conv_nd_zero_filter():
    x = torch.randn(1, 3, 16, 16)
    w = torch.zeros(6, 3, 3, 3)

    stride = (1, 1)
    padding = (0, 0)
    dilation = (1, 1)
    assert torch.allclose(
        fft_cross_correlation_nd(
            x, w, stride=stride, padding=padding, dilation=dilation
        ),
        F.conv2d(x, w, stride=stride, padding=padding, dilation=dilation),
        atol=1e-6,
    )


def test_fft_conv_nd_ones():
    x = torch.ones(1, 2, 10, 10)
    w = torch.ones(4, 2, 3, 3)

    stride = (2, 3)
    padding = (0, 0)
    dilation = (1, 1)

    assert torch.allclose(
        fft_cross_correlation_nd(
            x, w, stride=stride, padding=padding, dilation=dilation
        ),
        F.conv2d(x, w, stride=stride, padding=padding, dilation=dilation),
        atol=1e-4,
        rtol=1e-4,
    )


def test_fft_conv_nd_debug_stride():
    x = torch.randn(1, 1, 20, 20)
    w = torch.randn(1, 1, 3, 3)

    stride = (2, 3)
    padding = (0, 4)
    dilation = (2, 1)

    fft_out = fft_cross_correlation_nd(
        x, w, stride=stride, padding=padding, dilation=dilation
    )
    torch_out = F.conv2d(x, w, stride=stride, padding=padding, dilation=dilation)

    assert fft_out.shape == torch_out.shape
    assert torch.allclose(fft_out, torch_out, atol=1e-4, rtol=1e-4)
