import math
from copy import deepcopy
from typing import Literal, overload, override

from torch import Tensor, nn

from tarp.model.backbone.core import Encoder
from tarp.model.layers.attention.multihead import (
    MultiHeadSelfAttentionWithPositionalEncoding,
)
from tarp.model.layers.perceptron.gated import SwishGatedLinearUnitFeedForward
from tarp.model.layers.pooling.learned import SelfAttentionPooling
from tarp.model.layers.positional.core import AttentionBiasPositionalEncoding


class TransformerEncoderLayerWithPositionalEncoding(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        number_of_heads: int,
        feed_forward_dimension: int,
        positional_encoder: AttentionBiasPositionalEncoding,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        self.self_attention = MultiHeadSelfAttentionWithPositionalEncoding(
            model_dimension=model_dimension,
            number_of_heads=number_of_heads,
            positional_encoder=positional_encoder,
            dropout=dropout,
            bias=bias,
        )
        self.feed_forward = SwishGatedLinearUnitFeedForward(
            input_dimension=model_dimension,
            output_dimension=model_dimension,
            hidden_dimension=feed_forward_dimension,
            bias=bias,
        )
        self.attention_normalization = nn.RMSNorm(model_dimension)
        self.feedforward_normalization = nn.RMSNorm(model_dimension)
        self.dropout = nn.Dropout(dropout)

        self.scale = 1.0 / math.sqrt(2.0)

    @override
    def forward(
        self,
        features: Tensor,
        *,
        attention_mask: Tensor | None = None,
        positions: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        features = (
            features
            + self.dropout(
                self.self_attention(
                    self.attention_normalization(features),
                    attention_mask=attention_mask,
                    positions=positions,
                    is_causal=is_causal,
                )
            )
        ) * self.scale
        features = features + self.dropout(
            self.feed_forward(self.feedforward_normalization(features))
        )
        return features


class TransformerEncoder(Encoder):
    def __init__(
        self,
        model_dimension: int,
        number_of_heads: int,
        feed_forward_dimension: int,
        number_of_layers: int,
        positional_encoder: AttentionBiasPositionalEncoding,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        self.model_dimension = model_dimension
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerWithPositionalEncoding(
                    model_dimension=model_dimension,
                    number_of_heads=number_of_heads,
                    feed_forward_dimension=feed_forward_dimension,
                    positional_encoder=deepcopy(positional_encoder),
                    dropout=dropout,
                    bias=bias,
                )
                for _ in range(number_of_layers)
            ]
        )
        self.normalization = nn.RMSNorm(model_dimension)
        self.pooling = SelfAttentionPooling(feature_dimension=model_dimension)

    @overload
    def encode(
        self,
        sequence_embeddings: Tensor,
        attention_mask: Tensor,
        *,
        payload_mask: Tensor | None = None,
        positions: Tensor | None = None,
        mode: Literal["sequence"],
    ) -> Tensor: ...
    @overload
    def encode(
        self,
        sequence_embeddings: Tensor,
        attention_mask: Tensor,
        *,
        payload_mask: Tensor | None = None,
        positions: Tensor | None = None,
        mode: Literal["pooled"],
    ) -> Tensor: ...
    @overload
    def encode(
        self,
        sequence_embeddings: Tensor,
        attention_mask: Tensor,
        *,
        payload_mask: Tensor | None = None,
        positions: Tensor | None = None,
        mode: Literal["both"],
    ) -> tuple[Tensor, Tensor]: ...
    @override
    def encode(
        self,
        sequence_embeddings: Tensor,
        attention_mask: Tensor,
        *,
        payload_mask: Tensor | None = None,
        positions: Tensor | None = None,
        mode: Literal["sequence", "pooled", "both"],
    ) -> Tensor | tuple[Tensor, Tensor]:
        features = sequence_embeddings
        for layer in self.layers:
            features = layer(
                features,
                attention_mask=attention_mask.unsqueeze(1).unsqueeze(2),
                positions=positions,
                is_causal=False,
            )
        features = self.normalization(features)

        match mode:
            case "sequence":
                return features
            case "pooled":
                return self.pooling(features, attention_mask)
            case "both":
                pooled_features = self.pooling(
                    features, attention_mask, return_attention=False
                )
                return features, pooled_features

    @property
    @override
    def encoding_size(self) -> int:
        return self.model_dimension
