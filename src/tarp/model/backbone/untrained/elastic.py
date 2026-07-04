from copy import deepcopy
from typing import Literal, cast, final, overload, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tarp.model.backbone.core import Encoder
from tarp.model.backbone.untrained.transformer import (
    TransformerEncoder,
)
from tarp.model.layers.attention.windowed import WindowedCrossAttention
from tarp.model.layers.perceiver.elastic import (
    ElasticOptimalTransportPerceiver,
    ElasticOptimalTransportPerceiverOutput,
)
from tarp.model.layers.positional.core import AttentionBiasPositionalEncoding


@final
class ElasticPerceivedTransformerEncoder(Encoder):
    def __init__(
        self,
        model_dimension: int,
        number_of_heads: int,
        number_of_layers: int,
        feed_forward_dimension: int,
        positional_encoder: AttentionBiasPositionalEncoding,
        resolution: float = 1 / 3,
        window_radius: int = 4,
        temperature: float = 0.1,
        overflow_threshold: float = 1.0,
        iterations: int = 6,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        self.model_dimension = model_dimension
        self.resolution = resolution
        self.transformer = TransformerEncoder(
            model_dimension=model_dimension,
            number_of_heads=number_of_heads,
            number_of_layers=number_of_layers,
            feed_forward_dimension=feed_forward_dimension,
            positional_encoder=deepcopy(positional_encoder),
            dropout=dropout,
            bias=bias,
        )
        self.perceiver = ElasticOptimalTransportPerceiver(
            model_dimension=model_dimension,
            window_radius=window_radius,
            temperature=temperature,
            overflow_threshold=overflow_threshold,
            iterations=iterations,
            bias=bias,
            dropout=dropout,
        )
        self.cross_attention = WindowedCrossAttention(
            model_dimension=model_dimension,
            number_of_heads=number_of_heads,
            bias=bias,
            dropout=dropout,
        )
        self.normalization = nn.RMSNorm(model_dimension)
        self.last_metrics: dict[str, float] = {}

    @property
    @override
    def encoding_size(self) -> int:
        return self.model_dimension

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
        latent_lengths = (
            torch.ceil(attention_mask.sum(dim=1) * self.resolution).clamp_min(1).long()
        )
        latent_size = max(1, int(attention_mask.size(1) * self.resolution))
        (
            latent_tokens,
            latent_mask,
            latent_positions,
            latent_density,
            transport_plan,
            log_transport_plan,
            window_indices,
        ) = cast(
            ElasticOptimalTransportPerceiverOutput,
            self.perceiver(
                sequence_embeddings=sequence_embeddings,
                attention_mask=attention_mask,
                latent_size=latent_size,
                latent_lengths=latent_lengths,
            ),
        )

        with torch.no_grad():
            reconstructed_anchor = self.perceiver.reconstruct(
                latent_tokens=latent_tokens,
                transport_plan=transport_plan,
                window_indices=window_indices,
            )
            self.last_metrics["similarity"] = (
                F.cosine_similarity(
                    reconstructed_anchor * attention_mask.unsqueeze(-1),
                    sequence_embeddings.detach() * attention_mask.unsqueeze(-1),
                )
                .mean()
                .item()
            )

        latent_tokens, auxiliary_loss = self.transformer(
            sequence_embeddings=latent_tokens,
            attention_mask=latent_mask,
            positions=latent_positions,
            mode="sequence",
        )

        match mode:
            case "sequence":
                reconstructed = self.perceiver.reconstruct(
                    latent_tokens=latent_tokens,
                    transport_plan=transport_plan,
                    window_indices=window_indices,
                )
                decoded = self.cross_attention(
                    query=reconstructed,
                    key=latent_tokens,
                    value=None,
                    attention_bias=log_transport_plan,
                    routing_indices=window_indices,
                    window_mask=transport_plan > 0,
                )
                return self.normalization(decoded + reconstructed), auxiliary_loss

            case "pooled":
                return self.transformer.pooling(
                    latent_tokens, latent_mask
                ) + self.perceiver.pool(
                    latent_tokens=latent_tokens,
                    latent_mask=latent_mask,
                    latent_density=latent_density,
                ), auxiliary_loss
            case "both":
                reconstructed = self.perceiver.reconstruct(
                    latent_tokens=latent_tokens,
                    transport_plan=transport_plan,
                    window_indices=window_indices,
                )
                decoded = self.cross_attention(
                    query=reconstructed,
                    key=latent_tokens,
                    value=None,
                    attention_bias=log_transport_plan,
                    routing_indices=window_indices,
                    window_mask=transport_plan > 0,
                )
                pooled = self.transformer.pooling(
                    latent_tokens, latent_mask
                ) + self.perceiver.pool(
                    latent_tokens=latent_tokens,
                    latent_mask=latent_mask,
                    latent_density=latent_density,
                )
                return (
                    self.normalization(reconstructed + decoded),
                    pooled,
                    auxiliary_loss,
                )
