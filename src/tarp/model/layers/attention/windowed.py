import math
from typing import final, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@final
class WindowedCrossAttention(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        number_of_heads: int,
        bias: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_dimension = model_dimension
        self.number_of_heads = number_of_heads
        self.bias = bias
        self.dropout = dropout

        self.head_dimension = model_dimension // number_of_heads
        self.scale = 1.0 / math.sqrt(self.head_dimension)

        self.q_projection = nn.Linear(model_dimension, model_dimension, bias=bias)
        self.kv_projection = nn.Linear(model_dimension, 2 * model_dimension, bias=bias)
        self.output_projection = nn.Linear(model_dimension, model_dimension, bias=bias)

    @override
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor | None,
        *,
        attention_bias: Tensor,
        routing_indices: Tensor,
        window_mask: Tensor,
    ) -> Tensor:
        """
        :param Tensor query: [B, L_q, D] (B, L, D)
        :param Tensor key: [B, L_k, D] (B, T, D)
        :param Tensor value: Not used in cross-attention, included for API consistency. Should be None.
        :param Tensor attention_bias: [B, L_q, W] Attention bias for each query and its corresponding windowed keys (e.g., log of transport plan)
        :param Tensor routing_indices: [B, L_q, W] Indices of the keys that each query attends to (e.g., window indices)
        :param Tensor window_mask: [B, L_q, W] Mask indicating valid attention positions (e.g., routing mask)
        """
        batch_size, query_length, _ = query.shape
        _, key_length, _ = key.shape

        device = query.device
        negative_infinity = torch.finfo(attention_bias.dtype).min

        projected_queries = self.q_projection(query).reshape(
            batch_size, query_length, self.number_of_heads, self.head_dimension
        )  # [B, Q, H, D_h]

        projected_keys, projected_values = (
            self.kv_projection(key)
            .reshape(
                batch_size, key_length, 2, self.number_of_heads, self.head_dimension
            )
            .unbind(dim=2)
        )  # 2x [B, T, H, D_h]

        batch_indices = torch.arange(batch_size, device=device).reshape(
            batch_size, 1, 1
        )  # [B, 1, 1]

        windowed_keys = projected_keys[
            batch_indices, routing_indices
        ]  # [B, Q, W, H, D_h]
        windowed_values = projected_values[
            batch_indices, routing_indices
        ]  # [B, Q, W, H, D_h]

        # Multihead attention scores with scaling.
        # [B, Q, H, D_h] * [B, Q, W, H, D_h] -> [B, Q, H, W]
        scores = (
            torch.einsum("bqhd,bqwhd->bqhw", projected_queries, windowed_keys)
            * self.scale
        )  # [B, Q, H, W]

        scores = scores + attention_bias.unsqueeze(
            2
        )  # [B, Q, H, W] + [B, Q, W] -> [B, Q, H, W]

        masked_scores = scores.masked_fill(
            ~window_mask.bool().unsqueeze(2), negative_infinity
        )

        weights = F.softmax(masked_scores, dim=-1)

        weights = weights * window_mask.unsqueeze(2)

        # Apply dropout to attention weights
        weights = F.dropout(weights, p=self.dropout, training=self.training)

        context = torch.einsum(
            "bqhw,bqwhd->bqhd", weights, windowed_values
        )  # [B, Q, H, W] * [B, Q, W, H, D_h] -> [B, Q, H, D_h]

        context = context.reshape(
            batch_size, query_length, self.model_dimension
        )  # [B, Q, D]

        return self.output_projection(context)  # [B, Q, D]
