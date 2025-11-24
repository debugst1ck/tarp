# Utility functions for tensor operations
from typing import Union

from numpy.typing import NDArray
from torch import Tensor


def rescale(
    input: Union[Tensor, NDArray], minimum: float, maximum: float
) -> Union[Tensor, NDArray]:
    """
    Rescales the input tensor or ndarray linearly to a specified range [minimum, maximum].

    :param input: The input tensor or ndarray to be rescaled.
    :param minimum: The minimum value of the target range.
    :param maximum: The maximum value of the target range.
    :return: The rescaled tensor or ndarray.
    """
    normalized = (input - input.min()) / (input.max() - input.min())
    rescaled = normalized * (maximum - minimum) + minimum
    return rescaled
