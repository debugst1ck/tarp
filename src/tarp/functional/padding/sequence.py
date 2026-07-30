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

    maximum_length = max(sequence.size(0) for sequence in sequences)

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

    # Make new sequences list with dummy tensor appended
    dummy_sequences = sequences + [dummy_tensor]
    padded = pad_sequence(
        dummy_sequences,
        batch_first=batch_first,
        padding_value=padding_value,
        padding_side=padding_side,
    )

    return padded[:-1] if batch_first else padded[:, :-1]


def pad_to_length(
    sequences: list[Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
    padding_side: str = "right",
    length: int = 1024,
):
    if not sequences:
        return torch.empty(0)

    # Truncate first
    sequences = [sequence[:length] for sequence in sequences]

    maximum_length = max(sequence.size(0) for sequence in sequences)

    if maximum_length == length:
        return pad_sequence(
            sequences,
            batch_first=batch_first,
            padding_value=padding_value,
            padding_side=padding_side,
        )

    first_tensor = sequences[0]
    dummy_shape = (length,) + first_tensor.shape[1:]

    dummy_tensor = torch.full(
        dummy_shape,
        padding_value,
        dtype=first_tensor.dtype,
        device=first_tensor.device,
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
