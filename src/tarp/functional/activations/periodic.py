import torch
from torch import Tensor


def sine(input: Tensor, angular_frequency: float = 1.0) -> Tensor:
    """Applies a frequency-scaled sinusoidal transformation element-wise."""
    return torch.sin(angular_frequency * input)
