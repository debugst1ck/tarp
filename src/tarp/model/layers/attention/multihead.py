from collections.abc import Callable
from typing import override

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import (
    and_masks,
    create_block_mask,
    flex_attention,
)

from tarp.model.layers.positional.core import (
    NoPositionalEncoding,
    TransformativePositionalEncoding,
)


class SelfAttention(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        number_of_heads: int,
        positional_encoder: TransformativePositionalEncoding | None = None,
        bias: bool = False,
    ):
        super().__init__()
        self.model_dimension = model_dimension
        self.number_of_heads = number_of_heads
        self.head_dimension = model_dimension // number_of_heads

        self.positional_encoder = positional_encoder or NoPositionalEncoding()

        self.qkv_projection = nn.Linear(model_dimension, 3 * model_dimension, bias=bias)
        self.output_projection = nn.Linear(model_dimension, model_dimension, bias=bias)

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
        attention_mask: Tensor,
        positions: Tensor | None = None,
        score_mod_function: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]
        | None = None,
        mask_mod_function: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
        | None = None,
    ) -> Tensor:
        batch_size, sequence_length, _ = query.size()

        qkv = self.qkv_projection(query)
        qkv = qkv.reshape(
            batch_size, sequence_length, 3, self.number_of_heads, self.head_dimension
        )  # [B, L, 3, H, D_h]
        queries, keys, values = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # [B, H, L, D_h]

        positions = (
            positions
            if positions is not None
            else torch.arange(sequence_length, device=query.device)
            .unsqueeze(0)
            .expand(batch_size, sequence_length)
        )

        queries = self.positional_encoder(queries, positions)
        keys = self.positional_encoder(keys, positions)

        def padding_mask(
            batch: Tensor, head: Tensor, q_idx: Tensor, kv_idx: Tensor
        ) -> Tensor:
            return (
                attention_mask[batch, kv_idx].bool()
                & attention_mask[batch, q_idx].bool()
            )

        if mask_mod_function is not None:
            composite_mask_mod = and_masks(padding_mask, mask_mod_function)
        else:
            composite_mask_mod = padding_mask

        block_mask = create_block_mask(
            composite_mask_mod,
            B=batch_size,
            H=self.number_of_heads,
            Q_LEN=sequence_length,
            KV_LEN=sequence_length,
            device=query.device,
        )  # [B, H, Q_BLOCKS, KV_BLOCKS]

        attended = flex_attention(
            query=queries,
            key=keys,
            value=values,
            score_mod=score_mod_function,
            block_mask=block_mask,
        )  # [B, H, L, D_h]

        attended = attended.transpose(1, 2).reshape(
            batch_size, sequence_length, self.model_dimension
        )  # [B, L, D]
        output = self.output_projection(attended)  # [B, L, D]

        return output


class CausalSelfAttention(SelfAttention):
    @override
    def forward(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        *,
        attention_mask: Tensor,
        positions: Tensor | None = None,
        score_mod_function: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]
        | None = None,
        mask_mod_function: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
        | None = None,
    ) -> Tensor:
        def causal_mask(
            batch: Tensor, head: Tensor, q_idx: Tensor, kv_idx: Tensor
        ) -> Tensor:
            return q_idx >= kv_idx

        if mask_mod_function is not None:
            composite_mask_mod = and_masks(causal_mask, mask_mod_function)
        else:
            composite_mask_mod = causal_mask

        return super().forward(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            positions=positions,
            score_mod_function=score_mod_function,
            mask_mod_function=composite_mask_mod,
        )


class LegacySelfAttention(nn.Module):
    """
    Standard Self-Attention using F.scaled_dot_product_attention.
    """

    def __init__(
        self,
        model_dimension: int,
        number_of_heads: int,
        positional_encoder: TransformativePositionalEncoding | None = None,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model_dimension = model_dimension
        self.number_of_heads = number_of_heads
        self.head_dimension = model_dimension // number_of_heads
        self.dropout = dropout

        self.positional_encoder = positional_encoder or NoPositionalEncoding()

        self.qkv_projection = nn.Linear(model_dimension, 3 * model_dimension, bias=bias)
        self.output_projection = nn.Linear(model_dimension, model_dimension, bias=bias)

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
        attention_mask: Tensor,
        positions: Tensor | None = None,
    ) -> Tensor:
        batch_size, sequence_length, _ = query.size()

        qkv = self.qkv_projection(query)
        qkv = qkv.reshape(
            batch_size, sequence_length, 3, self.number_of_heads, self.head_dimension
        )
        queries, keys, values = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # [B, H, L, D_h]

        positions = (
            positions
            if positions is not None
            else torch.arange(sequence_length, device=query.device)
            .unsqueeze(0)
            .expand(batch_size, sequence_length)
        )

        queries = self.positional_encoder(queries, positions)
        keys = self.positional_encoder(keys, positions)

        headed_attention_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)

        attended = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=headed_attention_mask,
            is_causal=False,
            dropout_p=self.dropout if self.training else 0.0,
        )  # [B, H, L, D_h]

        attended = attended.transpose(1, 2).reshape(
            batch_size, sequence_length, self.model_dimension
        )
        return self.output_projection(attended)
