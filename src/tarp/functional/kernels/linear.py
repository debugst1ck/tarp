import torch
from torch import Tensor


def epanechnikov(distance: Tensor, bandwidth: Tensor) -> Tensor:
    return torch.clamp(1 - (distance / bandwidth).square(), min=0.0)


def quartic(distance: Tensor, bandwidth: Tensor) -> Tensor:
    return torch.clamp((1.0 - (distance / bandwidth).square()).square(), min=0.0)


def triweight(distance: Tensor, bandwidth: Tensor) -> Tensor:
    return torch.clamp((1.0 - (distance / bandwidth).square()).pow(3), min=0.0)


def gaussian(distance: Tensor, bandwidth: Tensor) -> Tensor:
    return torch.exp(-(1 / 2) * (distance / bandwidth).square())
