import math
from typing import override

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tarp.model.layers.positional.core import AttentionBiasPositionalEncoding


class WindowedCrossAttentionWithPositionalEncoding(nn.Module):
    """
    Abstract base class for Windowed and Routed Cross-Attention mechanisms.

    Subclasses handle how keys and values are fetched, aggregated, or structured
    """

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
        self.scale = math.sqrt(self.head_dimension)

        self.q_projection = nn.Linear(model_dimension, model_dimension, bias=bias)
        self.kv_projection = nn.Linear(model_dimension, 2 * model_dimension, bias=bias)
        self.output_projection = nn.Linear(model_dimension, model_dimension, bias=bias)

        self.positional_encoder = positional_encoder
        self.reset_parameters()

    def reset_parameters(self):
        self.q_projection.reset_parameters()
        self.kv_projection.reset_parameters()
        self.output_projection.reset_parameters()

    @override
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor | None = None,
        *,
        attention_bias: Tensor,
        routing_indices: Tensor,
        window_mask: Tensor,
        query_positions: Tensor,
        key_positions: Tensor,
    ) -> Tensor:
        """
        Forward pass for the attention mechanism.

        :param query: Tensor of shape (batch_size, query_length, model_dimension)
        :param key: Tensor of shape (batch_size, key_length, model_dimension)
        :param value: Optional Tensor of shape (batch_size, key_length, model_dimension). Not used.
        :param attention_bias: Tensor of shape (batch_size, query_length, window_size) containing attention bias values.
        :param window_mask: Tensor of shape (batch_size, query_length, window_size) containing boolean masks for valid attention positions.
        :param routing_indices: Optional Tensor of shape (batch_size, query_length, window_size) containing routing indices for attention.
        :param query_positions: Optional Tensor of shape (batch_size, query_length) containing positional indices for queries.
        :param key_positions: Optional Tensor of shape (batch_size, key_length) containing positional indices for keys.
        """
        batch_size, query_length, _ = query.shape
        _, key_length, _ = key.shape
        _, _, window_size = attention_bias.shape

        projected_queries = self.q_projection(query).reshape(
            batch_size, query_length, self.number_of_heads, self.head_dimension
        )  # [B, Q, H, D_h]

        projected_keys, projected_values = (
            self.kv_projection(key)
            .reshape(
                batch_size,
                key_length,
                2,
                self.number_of_heads,
                self.head_dimension,
            )
            .unbind(dim=2)
        )  # [B, K, H, D_h]

        positioned_query, positioned_key, _ = self.positional_encoder(
            projected_queries.permute(0, 2, 1, 3),
            projected_keys.permute(0, 2, 1, 3),
            query_positions=query_positions,
            key_positions=key_positions,
        )

        projected_queries = positioned_query.permute(0, 2, 1, 3)  # [B, Q, H, D_h]
        projected_keys = positioned_key.permute(0, 2, 1, 3)  # [B, K, H, D_h]

        batch_offsets = (
            torch.arange(
                batch_size, device=query.device, dtype=routing_indices.dtype
            ).reshape(batch_size, 1, 1)
            * key_length
        )  # [B, 1, 1]

        flat_indices = (batch_offsets + routing_indices).reshape(-1)  # [B*Q*W]

        flat_keys = projected_keys.reshape(
            batch_size * key_length,
            self.number_of_heads,
            self.head_dimension,
        )  # [B*K, H, D_h]

        flat_values = projected_values.reshape(
            batch_size * key_length,
            self.number_of_heads,
            self.head_dimension,
        )  # [B*K, H, D_h]

        windowed_keys = flat_keys[flat_indices].reshape(
            batch_size,
            query_length,
            window_size,
            self.number_of_heads,
            self.head_dimension,
        )  # [B, Q, W, H, D_h]

        windowed_values = flat_values[flat_indices].reshape(
            batch_size,
            query_length,
            window_size,
            self.number_of_heads,
            self.head_dimension,
        )  # [B, Q, W, H, Dh]

        # Query: [B, Q, H, Dh] x Windowed Keys: [B, Q, W, H, Dh] -> [B, Q, H, W]
        scores = (
            torch.einsum("bqhd,bqwhd->bqhw", projected_queries, windowed_keys)
            / self.scale
        )  # [B, Q, H, W]

        # Add attention bias to the scores
        scores = scores + attention_bias.unsqueeze(2)  # [B, Q, H, W]

        negative_infinity = torch.finfo(scores.dtype).min
        epsilon = torch.finfo(scores.dtype).eps

        scores = scores.masked_fill(
            ~window_mask.unsqueeze(2), negative_infinity
        )  # [B, Q, H, W]

        weights = F.softmax(scores, dim=-1) * window_mask.unsqueeze(2).to(scores.dtype)
        weights = weights / (weights.sum(dim=-1, keepdim=True)).clamp_min(epsilon)
        weights = F.dropout(
            weights, p=self.dropout, training=self.training
        )  # [B, Q, H, W]

        context = torch.einsum("bqhw,bqwhd->bqhd", weights, windowed_values).reshape(
            batch_size, query_length, self.model_dimension
        )  # [B, Q, D]

        return self.output_projection(context)  # [B, Q, D]
