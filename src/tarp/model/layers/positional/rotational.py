import torch
from torch import Tensor, nn


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding module.

    Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding
    (https://arxiv.org/abs/2104.09864)
    """

    def __init__(
        self,
        head_dimension: int,
        rotational_fraction: float = 1.0,
        maximum_sequence_length: int = 1024,
        base: int = 10000,
    ):
        """
        :param head_dimension: Dimension of the attention heads (embedding_dimension / number_of_heads)
        :param maximum_sequence_length: Maximum sequence length to precompute the embeddings for
        :param base: Base for the frequency calculation
        """
        super().__init__()
        self.dimension = head_dimension
        self.maximum_sequence_length = maximum_sequence_length
        self.base = base
        self.rotational_fraction = rotational_fraction

        self.rotary_dimension = int(self.dimension * self.rotational_fraction)

        assert self.dimension % 2 == 0, "Head dimension must be even."
        assert self.rotary_dimension % 2 == 0, "Rotary dimension must be even."

        # Declare buffers for precomputed values
        self.inverse_frequencies: Tensor
        self.cosine_cache: Tensor
        self.sine_cache: Tensor

        # Register buffers
        self.register_buffer(
            "inverse_frequencies",
            torch.zeros(self.rotary_dimension // 2),
            persistent=False,
        )
        self.register_buffer(
            "cosine_cache",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "sine_cache",
            torch.empty(0),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self):
        target_device = self.inverse_frequencies.device
        # Recompute 1 / base^(2i/d)
        inverse_frequencies = 1.0 / (
            self.base
            ** (
                torch.arange(
                    0,
                    self.rotary_dimension,
                    2,
                    dtype=torch.float,
                    device=target_device,
                )
                / self.rotary_dimension
            )
        )

        with torch.no_grad():
            # Object of type "Tensor" is not callable
            self.inverse_frequencies.copy_(inverse_frequencies)

        # Recompute trigonometric caches
        self._compute_trigonometric_caches(self.maximum_sequence_length)

    def _compute_trigonometric_caches(self, sequence_length: int):
        target_device = self.inverse_frequencies.device
        positions: Tensor = torch.arange(
            sequence_length, dtype=torch.float, device=target_device
        )  # (sequence_length,)
        angle_rates = torch.einsum("i,j->ij", positions, self.inverse_frequencies)

        # Apply rotation formula, Euler's formula: exp(i*angle) = cos(angle) + i*sin(angle)
        # Store with shape (1, 1, sequence_length, head_dimension/2) for broadcasting
        self.cosine_cache = angle_rates.cos().unsqueeze(0).unsqueeze(0)
        self.sine_cache = angle_rates.sin().unsqueeze(0).unsqueeze(0)

    def _apply_partial_rotary_embedding(
        self, input: Tensor, cosine: Tensor, sine: Tensor, rotary_dimension: int
    ) -> Tensor:
        """
        Apply rotary embedding to a fraction of the head dimension.

        :param Tensor input: Input tensor of shape (..., head_dimension)
        :param Tensor cosine: Cosine values of shape (..., head_dimension/2)
        :param Tensor sine: Sine values of shape (..., head_dimension/2)
        :param int rotary_dimension: Dimension to apply rotary embedding to
        :return Tensor: Tensor after applying rotary embedding
        """
        # Split the input into rotary and non-rotary parts
        input_rotary = input[..., :rotary_dimension]  # (..., rotary_dimension)
        input_passive = input[
            ..., rotary_dimension:
        ]  # (..., head_dimension - rotary_dimension)

        # Split rotary part into even and odd indices
        input_even = input_rotary[..., 0::2]  # (..., rotary_dimension/2)
        input_odd = input_rotary[..., 1::2]  # (..., rotary_dimension/2)

        # Slice to rotary dimension / 2
        cosine_rotary = cosine[..., : rotary_dimension // 2]
        sine_rotary = sine[..., : rotary_dimension // 2]

        # Apply rotary transformation
        # Using the rotation formula:
        # x' = x * cos(theta) - y * sin(theta)
        # y' = x * sin(theta) + y * cos(theta)
        rotated_even = (input_even * cosine_rotary) - (input_odd * sine_rotary)
        rotated_odd = input_odd * cosine_rotary + input_even * sine_rotary

        # Interleave rotated odd and even parts back into rotary dimension
        output_rotated = torch.empty_like(input_rotary)
        output_rotated[..., 0::2] = rotated_even
        output_rotated[..., 1::2] = rotated_odd

        # Concatenate the rotated and passive parts
        output = torch.cat([output_rotated, input_passive], dim=-1)
        return output

    def rotate_query_or_key(self, input: Tensor):
        """
        Apply rotary positional embeddings to a single tensor (query or key).

        :param Tensor input: Input tensor of shape (batch_size, number_of_heads, sequence_length, head_dimension)
        :return Tensor: Rotated tensor
        """
        batch_size, number_of_heads, sequence_length, _ = input.shape
        # Compute caches if needed
        if (self.cosine_cache is None) or (
            self.cosine_cache.shape[2] < sequence_length
        ):
            self._compute_trigonometric_caches(sequence_length)

        # Shape caches for broadcasting:
        # cosine/sine: (1, 1, L, D/2)
        cosine = self.cosine_cache[:, :, :sequence_length, :]
        sine = self.sine_cache[:, :, :sequence_length, :]

        # Rotary expects (..., D)
        # input: (B, H, L, D)
        rotated = self._apply_partial_rotary_embedding(
            input, cosine, sine, self.rotary_dimension
        )

        return rotated

    def rotate_query_and_key(self, query: Tensor, key: Tensor):
        """
        Apply rotary positional embeddings to query and key tensors.

        :param Tensor query: Query tensor of shape (batch_size, number_of_heads, sequence_length, head_dimension)
        :param Tensor key: Key tensor of shape (batch_size, number_of_heads, sequence_length, head_dimension)
        :return Tuple[Tensor, Tensor]: Tuple of rotated query and key tensors
        """
        rotated_query = self.rotate_query_or_key(query)
        rotated_key = self.rotate_query_or_key(key)
        return rotated_query, rotated_key

    def rotate_input(self, input: Tensor) -> Tensor:
        """
        Apply rotary positional embeddings to the input tensor without head dimension split.

        :param Tensor input: Input tensor of shape (batch_size, sequence_length, dimension)
        :return Tensor: Rotated tensor with shape (batch_size, sequence_length, dimension)
        """
        batch_size, sequence_length, dimension = input.shape
        # Make sure head dimension is even

        # Compute caches if needed
        if (self.cosine_cache is None) or (
            self.cosine_cache.shape[2] < sequence_length
        ):
            self._compute_trigonometric_caches(sequence_length)

        # Shape caches for broadcasting:
        # Caches: (1, 1, L, D/2) -> (1, L, D/2)
        cosine = self.cosine_cache[0, 0, :sequence_length]  # (L, D/2)
        sine = self.sine_cache[0, 0, :sequence_length]  # (L, D/2)
        cosine = cosine.unsqueeze(0)  # (1, L, D/2)
        sine = sine.unsqueeze(0)  # (1, L, D/2)

        return self._apply_partial_rotary_embedding(
            input, cosine, sine, self.rotary_dimension
        )

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply rotary positional embeddings to the input tensor.

        :param Tensor input: Input tensor of shape (batch_size, sequence_length, dimension)
        :return Tensor: Rotated tensor
        """
        return self.rotate_input(input)
