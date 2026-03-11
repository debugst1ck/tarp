from typing import Optional

from torch import Tensor, nn


class TransformativeRelativePositionalEncoder(nn.Module):
    """
    Attention-based Positional Encoder module.
    Meant to be used in Multi-Head Attention layers to encode positional information into the attention mechanism.

    Concrete implementations could be:
    - Rotary positional embeddings (e.g., RoFormer-style)
    """

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        *,
        positions: Optional[Tensor] = None,
        sequence_length: Optional[int] = None,
        offset: int = 0,
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        :param query: Query tensor of shape (batch_size, number_of_heads, sequence_length, head_dimension)
        :param key: Key tensor of shape (batch_size, number_of_heads, sequence_length, head_dimension)
        :param positions: Optional positions tensor of shape (batch_size, sequence_length)
        :param sequence_length: Optional sequence length
        :param offset: Offset for position calculation
        :return: Tuple of (query, key, attention_bias) where:
            - query: Query tensor with positional encoding applied, shape (batch_size, number_of_heads, sequence_length, head_dimension)
            - key: Key tensor with positional encoding applied, shape (batch_size, number_of_heads, sequence_length, head_dimension)
            - attention_bias: Optional attention bias tensor to be added to attention scores, shape (batch_size, number_of_heads, sequence_length, sequence_length)
        """
        return query, key, None  # Placeholder for attention bias
