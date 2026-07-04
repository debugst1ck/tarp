from copy import deepcopy
from typing import Literal, final, overload, override

from torch import Tensor, nn

from tarp.model.backbone.core import Encoder
from tarp.model.layers.attention.multihead import (
    MultiHeadSelfAttentionWithPositionalEncoding,
)
from tarp.model.layers.perceptron.gated import SwishGatedLinearUnitFeedForward
from tarp.model.layers.pooling.atomic import GlobalAveragePooling1D
from tarp.model.layers.positional.core import AttentionBiasPositionalEncoding


@final
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

    @override
    def forward(
        self,
        features: Tensor,
        *,
        attention_mask: Tensor | None = None,
        positions: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        features = features + self.dropout(
            self.self_attention(
                self.attention_normalization(features),
                attention_mask=attention_mask,
                positions=positions,
                is_causal=is_causal,
            )
        )
        features = features + self.dropout(
            self.feed_forward(self.feedforward_normalization(features))
        )
        return features


@final
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
        self.pooling = GlobalAveragePooling1D()

    @overload
    def encode(
        self,
        sequence_embeddings: Tensor,
        attention_mask: Tensor,
        *,
        positions: Tensor | None = None,
        mode: Literal["sequence", "pooled"],
    ) -> tuple[Tensor, Tensor | None]: ...
    @overload
    def encode(
        self,
        sequence_embeddings: Tensor,
        attention_mask: Tensor,
        *,
        positions: Tensor | None = None,
        mode: Literal["both"],
    ) -> tuple[Tensor, Tensor, Tensor | None]: ...
    @override
    def encode(
        self,
        sequence_embeddings: Tensor,
        attention_mask: Tensor,
        *,
        positions: Tensor | None = None,
        mode: Literal["sequence", "pooled", "both"],
    ) -> tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor, Tensor | None]:
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
                return features, None
            case "pooled":
                return self.pooling(features, attention_mask), None
            case "both":
                pooled_features = self.pooling(features, attention_mask)
                return features, pooled_features, None

    @property
    @override
    def encoding_size(self) -> int:
        return self.model_dimension
