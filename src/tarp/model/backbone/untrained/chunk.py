from copy import deepcopy
from typing import Literal, final, overload, override

from torch import Tensor

from tarp.model.backbone.core import Encoder
from tarp.model.backbone.untrained.transformer import TransformerEncoder
from tarp.model.layers.convolution.extraction import AdaptiveReceptiveField1D
from tarp.model.layers.perceiver.chunk import (
    ChunkedDecoder1D,
    ChunkedEncoder1D,
)
from tarp.model.layers.positional.core import AttentionBiasPositionalEncoding


@final
class ChunkedPerceiverEncoder(Encoder):
    def __init__(
        self,
        model_dimension: int,
        number_of_heads: int,
        number_of_layers: int,
        feed_forward_dimension: int,
        positional_encoder: AttentionBiasPositionalEncoding,
        compression_level: int = 3,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        self.model_dimension = model_dimension
        self.feature_extraction = AdaptiveReceptiveField1D(
            model_dimension=model_dimension,
            kernel_sizes=(1, 3, 5, 7, 9),
        )
        self.transformer = TransformerEncoder(
            model_dimension=model_dimension,
            number_of_heads=number_of_heads,
            number_of_layers=number_of_layers,
            feed_forward_dimension=feed_forward_dimension,
            positional_encoder=deepcopy(positional_encoder),
            dropout=dropout,
            bias=bias,
        )
        self.encoder = ChunkedEncoder1D(
            model_dimension=model_dimension,
            window_radius=compression_level,
            bias=bias,
            dropout=dropout,
        )
        self.decoder = ChunkedDecoder1D(
            model_dimension=model_dimension,
            number_of_heads=number_of_heads,
            window_radius=compression_level,
            bias=bias,
            dropout=dropout,
        )

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
        sequence_embeddings = self.feature_extraction(
            features=sequence_embeddings,
            attention_mask=attention_mask,
        )  # [B, L, D]
        latent_tokens, latent_mask, latent_positions = self.encoder(
            features=sequence_embeddings,
            attention_mask=attention_mask,
        )  # [B, T, D], [B, T], [B, T]

        decoded_latents, auxillary_loss = self.transformer(
            sequence_embeddings=latent_tokens,
            attention_mask=latent_mask,
            positions=positions,
            mode="sequence",
        )  # [B, T, D]

        reconstructed = self.decoder(
            sequence_features=sequence_embeddings,
            latent_tokens=decoded_latents,
            attention_mask=attention_mask,
            latent_mask=latent_mask,
            latent_positions=latent_positions,
        )  # [B, L, D]

        match mode:
            case "sequence":
                return reconstructed, auxillary_loss

            case "pooled":
                return self.transformer.pooling(
                    decoded_latents, latent_mask
                ), auxillary_loss

            case "both":
                return (
                    reconstructed,
                    self.transformer.pooling(decoded_latents, latent_mask),
                    auxillary_loss,
                )
