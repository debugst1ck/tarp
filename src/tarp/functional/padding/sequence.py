import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


def blocked_pad_sequence(
    sequences: list[Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
    padding_side: str = "right",
    block_size: int = 128,
) -> Tensor:
    """
    Wrapper around PyTorch C-based pad_sequence that forces the padded dimension
    to be a multiple of `block_size` using a fast dummy tensor allocation.
    """
    if not sequences:
        return torch.empty(0)

    maximum_length = max(seq.size(0) for seq in sequences)

    padded_length = ((maximum_length + block_size - 1) // block_size) * block_size

    if maximum_length == padded_length:
        return pad_sequence(
            sequences,
            batch_first=batch_first,
            padding_value=padding_value,
            padding_side=padding_side,
        )

    first_tensor = sequences[0]
    dummy_shape = (padded_length,) + first_tensor.shape[1:]
    dummy_tensor = torch.full(
        dummy_shape, padding_value, dtype=first_tensor.dtype, device=first_tensor.device
    )

    sequences.append(dummy_tensor)
    padded = pad_sequence(
        sequences,
        batch_first=batch_first,
        padding_value=padding_value,
        padding_side=padding_side,
    )
    _ = sequences.pop()

    return padded[:-1] if batch_first else padded[:, :-1]
