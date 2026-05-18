import torch
from torch import Tensor


def log_cauchy(distances: Tensor, sharpness: float | Tensor) -> Tensor:
    """
    Logarithm of the Cauchy kernel function.
    """
    return -torch.log1p(sharpness * distances.square())


def log_gaussian(distances: Tensor, sharpness: float | Tensor) -> Tensor:
    return -sharpness * distances.square() / 2


def log_laplace(distances: Tensor, sharpness: float | Tensor) -> Tensor:
    return -sharpness * distances.abs()


def log_epanechnikov(distances: Tensor, sharpness: float | Tensor) -> Tensor:
    info = torch.finfo(distances.dtype)
    base = (1.0 - sharpness * distances.square()).clamp(min=info.tiny)
    return torch.where(base > 0.0, torch.log(base), (info.min / 2) * torch.abs(base))


def log_rational_power(distances: Tensor, sharpness: float | Tensor) -> Tensor:
    return -0.5 * torch.log1p(sharpness * distances.square())


def log_triweight(distances: Tensor, sharpness: float | Tensor) -> Tensor:
    info = torch.finfo(distances.dtype)
    base = (1.0 - sharpness * distances.square()).clamp(min=info.tiny)
    return torch.where(
        base > 0.0, 3.0 * torch.log(base), (info.min / 2) * torch.abs(base)
    )
