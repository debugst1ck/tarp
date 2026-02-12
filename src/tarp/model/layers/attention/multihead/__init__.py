from typing import Optional

import torch.nn.functional as F
from torch import Tensor, nn

from tarp.model.layers.positional import AttentionPositionalEncoder


class MultiHeadSelfAttentionWithPositionalEncoding(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        number_of_heads: int,
        positional_encoder: AttentionPositionalEncoder,
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

        self.positional_encoder = positional_encoder
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
        queries, keys = self.positional_encoder(queries, keys)

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
