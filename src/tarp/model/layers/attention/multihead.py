from typing import override

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tarp.model.layers.positional.core import AttentionBiasPositionalEncoding


class MultiHeadSelfAttentionWithPositionalEncoding(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        number_of_heads: int,
        positional_encoder: AttentionBiasPositionalEncoding,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        self.model_dimension = model_dimension
        self.number_of_heads = number_of_heads
        self.head_dimension = model_dimension // number_of_heads
        self.dropout = dropout

        # Linear projections for Q, K, V
        self.qkv_projection = nn.Linear(model_dimension, 3 * model_dimension, bias=bias)
        self.output_projection = nn.Linear(model_dimension, model_dimension, bias=bias)

        self.positional_encoder = positional_encoder
        self.reset_parameters()

    def reset_parameters(self):
        self.qkv_projection.reset_parameters()
        self.output_projection.reset_parameters()

    @override
    def forward(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        *,
        attention_mask: Tensor | None = None,
        positions: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        """
        :param Tensor query: Input tensor of shape [B, L, D]
        :param Tensor key: Optional key tensor of shape [B, L, D]. Not used in self-attention, but included for API consistency.
        :param Tensor value: Optional value tensor of shape [B, L, D]. Not used in self-attention, but included for API consistency.
        :param Tensor attention_mask: Optional attention mask of shape [B, 1, 1, L] or [B, 1, L, L]
        :param Tensor positions: Optional positions tensor of shape [B, L] containing position indices for each token in the sequence.
        :param bool is_causal: Whether to apply causal masking (prevent attending to future tokens)
        """
        batch_size, sequence_length, _ = query.size()

        qkv: Tensor = self.qkv_projection(query)
        qkv = qkv.view(
            batch_size, sequence_length, 3, self.number_of_heads, self.head_dimension
        )
        queries, keys, values = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, D_h]

        positions = (
            positions
            if positions is not None
            else torch.arange(sequence_length, device=query.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        queries, keys, positional_attention_bias = self.positional_encoder(
            queries, keys, query_positions=positions, key_positions=positions
        )

        attention_bias: Tensor | None = positional_attention_bias  # [B, H, L, L]
        if attention_bias is not None:
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
        else:
            attention_bias = attention_mask

        attended = F.scaled_dot_product_attention(
            query=queries,
            key=keys,
            value=values,
            attn_mask=attention_bias,
            is_causal=is_causal if attention_bias is None else False,
            dropout_p=self.dropout if self.training else 0.0,
        )  # [B, H, L, D_h]

        attended = attended.transpose(1, 2).reshape(
            batch_size, sequence_length, self.model_dimension
        )  # [B, L, D]

        return self.output_projection(attended)  # [B, L, D]
