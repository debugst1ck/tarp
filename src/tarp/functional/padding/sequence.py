import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


def blocked_pad_sequence(
    sequences: list[Tensor],
    padding_value: float = 0.0,
    block_size: int = 128,
) -> Tensor:
    """
    Pads a list of variable-length sequences to the nearest multiple of `block_size`.
    :param list[Tensor] sequences: List of variable-length sequences (1D or 2D tensors).
    :param float padding_value: Value to use for padding.
    :param int block_size: Block size to pad to (default: 128). Aligns to CUDA warp size for efficiency.
    :return Tensor: A tensor of shape [B, L, ...] where L is the nearest multiple of `block_size` greater than or equal to the maximum sequence length.
    """
    if not sequences:
        return torch.empty(0)

    padded = pad_sequence(
        sequences,
        batch_first=True,  # Hardcoded: modern standard
        padding_value=padding_value,
    )

    # 2. Determine current length and calculate block-aligned target length
    current_length = padded.size(1)  # Shape is [B, L, ...]
    padded_length = ((current_length + block_size - 1) // block_size) * block_size
    pad_amount = padded_length - current_length

    if pad_amount == 0:
        return padded

    return F.pad(padded, (0, 0, 0, pad_amount), value=padding_value)


def pad_to_length(
    sequences: list[Tensor],
    padding_value: float = 0.0,
    length: int = 1024,
) -> Tensor:
    """
    Truncates and pads a list of variable-length sequences to a fixed `length`.
    :param list[Tensor] sequences: List of variable-length sequences (1D or 2D tensors).
    :param float padding_value: Value to use for padding.
    :param int length: Target length to pad/truncate to (default: 1024).
    :return Tensor: A tensor of shape [B, length, ...] where B is the batch size and length is the specified target length.
    """
    if not sequences:
        return torch.empty(0)

    sequences = [seq[:length] for seq in sequences]

    padded = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=padding_value,
    )

    current_length = padded.size(1)
    pad_amount = length - current_length

    if pad_amount == 0:
        return padded

    return F.pad(padded, (0, 0, 0, pad_amount), value=padding_value)
