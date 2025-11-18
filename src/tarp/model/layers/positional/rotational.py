import torch
from torch import nn, Tensor

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding module.

    Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding
    (https://arxiv.org/abs/2104.09864)
    """

    def __init__(
        self,
        head_dimension: int,
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

        # Precompute the inverse
        inverse_frequencies = 1.0 / (
            base ** (torch.arange(0, head_dimension, 2).float() / head_dimension)
        )

        # Register as buffers
        self.register_buffer(
            "inverse_frequencies", inverse_frequencies, persistent=False
        )

        # Precompute the cosine and sine caches
        self.register_buffer("cosine_cache", None, persistent=False)
        self.register_buffer("sine_cache", None, persistent=False)

        # Initial computation of caches
        self._compute_trigonometric_caches(maximum_sequence_length)

    def _compute_trigonometric_caches(self, sequence_length: int):
        positions: Tensor = torch.arange(
            sequence_length, device=self.inverse_frequencies.device
        ).float()  # (sequence_length,)
        angle_rates = torch.einsum("i,j->ij", positions, self.inverse_frequencies)

        # Apply rotation formula, Euler's formula: exp(i*angle) = cos(angle) + i*sin(angle)
        # Store with shape (1, 1, sequence_length, head_dimension/2) for broadcasting
        self.cosine_cache = angle_rates.cos().unsqueeze(0).unsqueeze(0)
        self.sine_cache = angle_rates.sin().unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _apply_rotary_embedding(input: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
        """
        Apply rotary embedding to a given tensor.

        :param Tensor tensor: Input tensor of shape (..., head_dimension)
        :param Tensor cosine: Cosine values of shape (..., head_dimension/2)
        :param Tensor sine: Sine values of shape (..., head_dimension/2)
        :return Tensor: Tensor after applying rotary embedding
        """
        input_reshaped = input.reshape(
            *input.shape[:-1], -1, 2
        )  # (..., head_dimension/2, 2)

        # Apply rotation
        rotated = torch.stack(
            [
                input_reshaped[..., 0] * cosine - input_reshaped[..., 1] * sine,
                input_reshaped[..., 0] * sine + input_reshaped[..., 1] * cosine,
            ],
            dim=-1,
        )

        return rotated.reshape(input.shape)  # (..., head_dimension)

    def rotate_query_or_key(self, input: Tensor):
        """
        Apply rotary positional embeddings to a single tensor (query or key).

        :param Tensor input: Input tensor of shape (batch_size, number_of_heads, sequence_length, head_dimension)
        :return Tensor: Rotated tensor
        """

        batch_size, number_of_heads, sequence_length, _ = input.shape
        # Make sure head dimension is even
        assert (
            self.dimension % 2 == 0
        ), "Head dimension must be even for rotary embeddings."

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
        rotated = self._apply_rotary_embedding(input, cosine, sine)

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

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply rotary positional embeddings to the input tensor.

        :param Tensor input: Input tensor of shape (batch_size, sequence_length, number_of_heads, head_dimension)
        :return Tensor: Rotated tensor
        """
        return self.rotate_query_or_key(input)