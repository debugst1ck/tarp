import torch
from torch import Tensor


def dropout(
    input: Tensor,
    p: float = 0.5,
    value: float = 0.0,
    training: bool = True,
    in_place: bool = False,
) -> Tensor:
    """
    Dropout function that sets elements to a specified value instead of zero.

    :param Tensor input: The input tensor.
    :param float p: The probability of an element to be dropped. Default is 0.5.
    :param float value: The value to set for dropped elements. Default is 0.0.
    :param bool training: If True, applies dropout; otherwise, returns the input unchanged. Default is True.
    :param bool inplace: If True, modifies the input tensor in-place. Default is False.
    :return Tensor: The output tensor after applying dropout.
    """
    if not training or p == 0.0:
        return input

    if p == 1.0:
        return torch.full_like(input, value)

    mask = torch.rand_like(input) < p
    if in_place:
        return input.masked_fill_(mask, value)
    else:
        return input.masked_fill(mask, value)
