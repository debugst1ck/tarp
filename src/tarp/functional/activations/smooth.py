import torch
from torch import Tensor


def inverse_softplus(value: Tensor) -> Tensor:
    return torch.log(torch.expm1(value))
