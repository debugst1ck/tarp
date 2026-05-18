from torch import Tensor
from torch.nn import functional as F


def swiglu(value: Tensor, gate: Tensor) -> Tensor:
    """
    SwiGLU: Swish-Gated Linear Unit.

    :param Tensor value: The 'value' tensor (usually the second half of a linear projection).
    :param Tensor gate: The 'gate' tensor (usually the first half of a linear projection).
    :return: Swish(gate) * value
    """
    return F.silu(gate) * value


def reglu(value: Tensor, gate: Tensor) -> Tensor:
    """
    ReGLU: ReLU-Gated Linear Unit.

    :param Tensor value: The 'value' tensor (usually the second half of a linear projection).
    :param Tensor gate: The 'gate' tensor (usually the first half of a linear projection).
    :return: ReLU(gate) * value
    """
    return F.relu(gate) * value


def geglu(value: Tensor, gate: Tensor) -> Tensor:
    """
    GeGLU: Gaussian Error Gated Linear Unit.

    :param Tensor value: The 'value' tensor (usually the second half of a linear projection).
    :param Tensor gate: The 'gate' tensor (usually the first half of a linear projection).
    :return: GELU(gate) * value
    """
    return F.gelu(gate) * value
