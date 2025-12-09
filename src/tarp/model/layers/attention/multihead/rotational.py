from typing import Optional

import torch.nn.functional as F
from torch import Tensor, nn

from tarp.model.layers.positional.rotational import RotaryPositionalEmbedding


class MultiHeadSelfAttentionWithRotaryPositionalEmbeddings(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        number_of_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_dimension = model_dimension
        self.number_of_heads = number_of_heads
        self.head_dimension = model_dimension // number_of_heads
        self.dropout = dropout

        # Linear projections for Q, K, V
        self.qkv_projection = nn.Linear(model_dimension, 3 * model_dimension)
        self.output_projection = nn.Linear(model_dimension, model_dimension)

        self.rope = RotaryPositionalEmbedding(
            head_dimension=self.head_dimension, rotational_fraction=1.0
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.qkv_projection.reset_parameters()
        self.output_projection.reset_parameters()

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """
        :param Tensor query: Input query tensor of shape (batch_size, sequence_length, model_dimension)
        :param Tensor key: Not used in self-attention
        :param Tensor value: Not used in self-attention
        :param Tensor attention_mask: Optional attention mask of shape (batch_size, 1, 1, sequence_length)
        :param bool is_causal: Whether to apply causal masking
        """
        batch_size, sequence_length, _ = query.size()

        # Linear projections for query, key, value
        qkv: Tensor = self.qkv_projection(
            query
        )  # (batch_size, seq_len, 3*embedding_dimension)
        qkv = qkv.reshape(
            batch_size, sequence_length, 3, self.number_of_heads, self.head_dimension
        )  # (batch_size, seq_len, 3, number_of_heads, head_dimension)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # (3, batch_size, number_of_heads, seq_len, head_dimension)

        queries, keys, values = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # Each: (batch_size, number_of_heads, seq_len, head_dimension)

        # Apply RoPE to queries and keys
        queries, keys = self.rope.rotate_query_and_key(queries, keys)

        # Scaled dot-product attention Flash Attention
        attention_output = F.scaled_dot_product_attention(
            query=queries,
            key=keys,
            value=values,
            attn_mask=attention_mask,
            is_causal=is_causal and attention_mask is None,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # Transpose (batch_size, number_of_heads, seq_len, head_dimension) -> (batch_size, seq_len, embedding_dimension)
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, sequence_length, self.model_dimension
        )

        return self.output_projection(attention_output)


class MultiHeadCrossAttentionWithRotaryPositionalEmbeddings(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        number_of_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_dimension = model_dimension
        self.number_of_heads = number_of_heads
        self.head_dimension = model_dimension // number_of_heads
        self.dropout = dropout

        # Linear projections for Q, K, V
        self.q_projection = nn.Linear(model_dimension, model_dimension)
        self.k_projection = nn.Linear(model_dimension, model_dimension)
        self.v_projection = nn.Linear(model_dimension, model_dimension)

        self.output_projection = nn.Linear(model_dimension, model_dimension)

        self.rope = RotaryPositionalEmbedding(head_dimension=self.head_dimension)

        self.reset_parameters()

    def reset_parameters(self):
        self.q_projection.reset_parameters()
        self.k_projection.reset_parameters()
        self.v_projection.reset_parameters()
        self.output_projection.reset_parameters()

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """
        :param Tensor query: Input query tensor of shape (batch_size, query_length, model_dimension)
        :param Tensor key: Input key tensor of shape (batch_size, key_length, model_dimension)
        :param Tensor value: Input value tensor of shape (batch_size, value_length, model_dimension)
        :param Tensor attention_mask: Optional attention mask of shape (batch_size, key_length)
        :param bool is_causal: Whether to apply causal masking
        :return Tensor: Output tensor of shape (batch_size, query_length, model_dimension)
        :raises ValueError: If key or value is None
        """
        batch_size, query_length, _ = query.size()

        # Determine key or value length, assuming key and value have the same length
        key_or_value_length = key.size(1) if key is not None else value.size(1)

        # Linear projections for Q, K, V
        queries: Tensor = (
            self.q_projection(query)
            .reshape(
                batch_size, query_length, self.number_of_heads, self.head_dimension
            )
            .transpose(1, 2)
        )  # Each: (batch_size, number_of_heads, seq_len, head_dimension)

        keys: Tensor = (
            self.k_projection(key)
            .reshape(
                batch_size,
                key_or_value_length,
                self.number_of_heads,
                self.head_dimension,
            )
            .transpose(1, 2)
        )  # Each: (batch_size, number_of_heads, seq_len, head_dimension)

        values: Tensor = (
            self.v_projection(value)
            .reshape(
                batch_size,
                key_or_value_length,
                self.number_of_heads,
                self.head_dimension,
            )
            .transpose(1, 2)
        )  # Each: (batch_size, number_of_heads, seq_len, head_dimension)

        # Apply RoPE to queries and keys
        queries, keys = self.rope.rotate_query_and_key(queries, keys)

        if attention_mask is not None:
            # attention_mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            attention_mask = attention_mask[:, None, None, :].to(query.dtype)
            attention_mask = (1.0 - attention_mask) * -1e9  # convert to additive mask

        # Scaled dot-product attention Flash Attention
        attention_output = F.scaled_dot_product_attention(
            query=queries,
            key=keys,
            value=values,
            attn_mask=attention_mask,
            is_causal=is_causal and attention_mask is None,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # Transpose (batch_size, number_of_heads, seq_len, head_dimension) -> (batch_size, seq_len, model_dimension)
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, query_length, self.model_dimension
        )

        return self.output_projection(attention_output)
