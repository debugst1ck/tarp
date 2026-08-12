from torch import Tensor
from torch.nn import functional as F


def swiglu(gate: Tensor, value: Tensor) -> Tensor:
    """
    SwiGLU: Swish-Gated Linear Unit.

    :param Tensor gate: The 'gate' tensor (usually the first half of a linear projection).
    :param Tensor value: The 'value' tensor (usually the second half of a linear projection).
    :return: Swish(gate) * value
    """
    return F.silu(gate) * value


def reglu(gate: Tensor, value: Tensor) -> Tensor:
    """
    ReGLU: ReLU-Gated Linear Unit.

    :param Tensor gate: The 'gate' tensor (usually the first half of a linear projection).
    :param Tensor value: The 'value' tensor (usually the second half of a linear projection).
    :return: ReLU(gate) * value
    """
    return F.relu(gate) * value


def geglu(gate: Tensor, value: Tensor) -> Tensor:
    """
    GeGLU: Gaussian Error Gated Linear Unit.

    :param Tensor gate: The 'gate' tensor (usually the first half of a linear projection).
    :param Tensor value: The 'value' tensor (usually the second half of a linear projection).
    :return: GELU(gate) * value
    """
    return F.gelu(gate) * value
