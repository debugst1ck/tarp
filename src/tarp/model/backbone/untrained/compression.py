from copy import deepcopy
from typing import Literal, overload, override

from torch import Tensor, nn

from tarp.model.backbone.core import Encoder
from tarp.model.backbone.untrained.transformer import (
    TransformerEncoderLayerWithPositionalEncoding,
)
from tarp.model.layers.compression.transport import (
    ElasticKernelDensityCompression1D,
    ElasticKernelDensityCompressionOutput,
)
from tarp.model.layers.positional.core import AttentionBiasPositionalEncoding


class ElasticCompressedTransformerEncoder(Encoder):
    def __init__(
        self,
        model_dimension: int,
        number_of_heads: int,
        number_of_layers: int,
        feed_forward_dimension: int,
        positional_encoder: AttentionBiasPositionalEncoding,
        resolution: float = 0.5,
        locality_radius: int = 6,
        positional_weight: float = 0.4,
        background_cost_payload: float = 2.0,
        minimum_budget_usage: float = 0.5,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        self.model_dimension = model_dimension
        self.global_attention = nn.ModuleList(
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
        self.elastic_tokenizer = ElasticKernelDensityCompression1D(
            embedding_dimension=model_dimension,
            resolution=resolution,
            locality_radius=locality_radius,
            positional_weight=positional_weight,
            background_cost_payload=background_cost_payload,
            minimum_budget_usage=minimum_budget_usage,
            number_of_heads=number_of_heads,
            dropout=dropout,
            bias=bias,
        )
        self.normalization = nn.RMSNorm(model_dimension)

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
        output: ElasticKernelDensityCompressionOutput = self.elastic_tokenizer(
            sequence=sequence_embeddings,
            sequence_mask=attention_mask,
            payload_mask=payload_mask if payload_mask is not None else attention_mask,
        )
        features = output.tokens
        # Pass through global attention layers
        for layer in self.global_attention:
            features = layer(
                features=features,
                attention_mask=output.sink_mask.unsqueeze(1).unsqueeze(2),
                positions=output.sink_coordinates,
            )
        features = self.normalization(features)
        match mode:
            # Trust me bro
            case "sequence":
                return self.elastic_tokenizer.reconstruct(
                    features,
                    output.sequence_features,
                    output.window_sink_indices,
                    output.window_mask,
                    output.reconstruction_bias,
                    output.source_coordinates,
                    output.sink_coordinates,
                )
            case "pooled":
                return self.elastic_tokenizer.pooling(
                    features,
                    output.sink_mass,
                    output.sink_mask,
                )
            case "both":
                return self.elastic_tokenizer.reconstruct(
                    features,
                    output.sequence_features,
                    output.window_sink_indices,
                    output.window_mask,
                    output.reconstruction_bias,
                    output.source_coordinates,
                    output.sink_coordinates,
                ), self.elastic_tokenizer.pooling(
                    features,
                    output.sink_mass,
                    output.sink_mask,
                )

    @property
    @override
    def encoding_size(self) -> int:
        return self.model_dimension
