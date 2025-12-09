# Utility functions for tensor operations
import torch
from torch import Tensor


def rescale(
    input: Tensor, minimum: float, maximum: float, epsilon: float = 1e-8
) -> Tensor:
    """
    Rescales the input tensor linearly to a specified range [minimum, maximum].

    :param input: The input tensor to be rescaled.
    :param minimum: The minimum value of the target range.
    :param maximum: The maximum value of the target range.
    :param epsilon: A small value to prevent division by zero.
    :return: The rescaled tensor
    """

    # Mask invalid values
    finite_mask = torch.isfinite(input)
    finite_values = input[finite_mask]

    # Edge case: no finite values
    if finite_values.numel() == 0:
        return torch.full_like(input, (minimum + maximum) / 2)

    input_minimum = finite_values.min()
    input_maximum = finite_values.max()

    # Edge case where all values are the same
    # In this case, return a tensor filled with the midpoint of the target range
    if abs(float(input_maximum - input_minimum)) < epsilon:
        rescaled = torch.zeros_like(input) + (minimum + maximum) / 2

    scaled = (maximum - minimum) / (input_maximum - input_minimum + epsilon)
    rescaled = (input - input_minimum) * scaled + minimum

    # Clamp to the target range
    rescaled = torch.clamp(rescaled, minimum, maximum)
    return rescaled
