import torch
from torch import Tensor


def log_cauchy(distances: Tensor, bandwidth: Tensor) -> Tensor:
    """
    Logarithm of the Cauchy kernel function.
    """
    # Equivalent to: -log(1 + (distance / bandwidth)^2)
    return -torch.log1p(distances.square() / bandwidth.square())


def log_gaussian(distances: Tensor, bandwidth: Tensor) -> Tensor:
    """
    Logarithm of the Gaussian kernel function.
    """
    # Equivalent to: -0.5 * (distance / bandwidth)^2
    return -0.5 * (distances.square() / bandwidth.square())


def log_laplace(distances: Tensor, bandwidth: Tensor) -> Tensor:
    """
    Logarithm of the Laplace (exponential) kernel function.
    """
    # Equivalent to: -|distance| / bandwidth
    return -distances.abs() / bandwidth


def log_rational_power(distances: Tensor, bandwidth: Tensor) -> Tensor:
    """
    Logarithm of the Rational Quadratic kernel function.
    """
    return -0.5 * torch.log1p(distances.square() / bandwidth.square())


def log_epanechnikov(distances: Tensor, bandwidth: Tensor) -> Tensor:
    """
    Logarithm of the Epanechnikov parabolic boundary kernel function.
    """
    info = torch.finfo(distances.dtype)
    scaled_square = distances.square() / bandwidth.square()
    inside = scaled_square < 1.0
    return torch.where(
        inside,
        torch.log1p(-torch.where(inside, scaled_square, 0.0)),
        torch.full_like(scaled_square, info.min / 2),
    )


def log_quartic(distances: Tensor, bandwidth: Tensor) -> Tensor:
    """
    Logarithm of the smooth Quartic (Biweight) compact support boundary kernel.
    """
    info = torch.finfo(distances.dtype)
    scaled_square = distances.square() / bandwidth.square()
    inside = scaled_square < 1.0
    return torch.where(
        inside,
        2.0 * torch.log1p(-torch.where(inside, scaled_square, 0.0)),
        torch.full_like(scaled_square, info.min / 2),
    )


def log_triweight(distances: Tensor, bandwidth: Tensor) -> Tensor:
    """
    Logarithm of the smooth Triweight compact support boundary kernel.
    """
    info = torch.finfo(distances.dtype)
    scaled_square = distances.square() / bandwidth.square()
    inside = scaled_square < 1.0
    return torch.where(
        inside,
        3.0 * torch.log1p(-torch.where(inside, scaled_square, 0.0)),
        torch.full_like(scaled_square, info.min / 2),
    )
