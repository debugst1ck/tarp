from copy import deepcopy
from typing import Literal, final, overload, override

import torch
from torch import Tensor, nn

from tarp.model.backbone.core import Encoder
from tarp.model.backbone.untrained.transformer import (
    TransformerEncoderLayerWithPositionalEncoding,
)
from tarp.model.layers.perceiver.elastic import (
    ElasticOptimalTransportPerceiver,
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
        resolution: float = 0.5,
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
        self.perceiver = ElasticOptimalTransportPerceiver(
            model_dimension=model_dimension,
            window_radius=window_radius,
            temperature=temperature,
            overflow_threshold=overflow_threshold,
            iterations=iterations,
            bias=bias,
            dropout=dropout,
            hidden_dimension=feed_forward_dimension,
        )
        self.normalization = nn.RMSNorm(model_dimension)

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
        mode: Literal["sequence"],
    ) -> tuple[Tensor, Tensor | None]: ...
    @overload
    def encode(
        self,
        sequence_embeddings: Tensor,
        attention_mask: Tensor,
        *,
        positions: Tensor | None = None,
        mode: Literal["pooled"],
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
            window_indices,
        ) = self.perceiver(
            sequence_embeddings=sequence_embeddings,
            attention_mask=attention_mask,
            latent_size=latent_size,
            latent_lengths=latent_lengths,
        )

        for layer in self.layers:
            latent_tokens = layer(
                latent_tokens,
                attention_mask=torch.where(
                    latent_mask > 0,
                    torch.zeros_like(latent_mask),
                    torch.full_like(latent_mask, torch.finfo(latent_tokens.dtype).min),
                )
                .unsqueeze(1)
                .unsqueeze(2),
                positions=latent_positions,
                is_causal=False,
            )
        match mode:
            case "sequence":
                reconstructed = self.perceiver.reconstruct(
                    latent_tokens=latent_tokens,
                    transport_plan=transport_plan,
                    window_indices=window_indices,
                )
                return self.normalization(reconstructed), None

            case "pooled":
                return self.perceiver.pool(
                    latent_tokens=latent_tokens,
                    latent_mask=latent_mask,
                    latent_density=latent_density,
                ), None
            case "both":
                reconstructed = self.perceiver.reconstruct(
                    latent_tokens=latent_tokens,
                    transport_plan=transport_plan,
                    window_indices=window_indices,
                )

                return (
                    self.normalization(reconstructed),
                    self.perceiver.pool(
                        latent_tokens=latent_tokens,
                        latent_mask=latent_mask,
                        latent_density=latent_density,
                    ),
                    None,
                )
