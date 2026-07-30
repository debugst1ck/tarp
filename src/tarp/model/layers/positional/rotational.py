from abc import ABC, abstractmethod
from typing import final, override

import torch
from torch import Tensor, nn

from tarp.model.layers.positional.core import TransformativePositionalEncoding


class RotaryPositionalEncoding(TransformativePositionalEncoding, ABC):
    def __init__(
        self,
        dimension: int,
        rotational_fraction: float = 1.0,
        base: int = 10_000,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.dimension = dimension  # [D]
        self.rotational_fraction = rotational_fraction
        self.base = base
        self.rotary_dimension = int(dimension * rotational_fraction)  # [R]

        # The dtype for core calculations
        self.dtype = dtype

        self.inverse_frequencies = nn.Buffer(
            1.0
            / (
                self.base
                ** (
                    torch.arange(
                        0,
                        self.rotary_dimension,
                        2,
                        dtype=self.dtype,
                    )
                    / self.rotary_dimension
                )
            ),
            persistent=False,
        )

    @abstractmethod
    def trigonometric_position_frequencies(
        self, positions: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the sine and cosine components for the given positions.

        :param Tensor positions: [B, L] The positions for which to compute the rotations.
        """
        raise NotImplementedError

    def _rotate_half(self, features: Tensor) -> Tensor:
        half_dimension = features.shape[-1] // 2
        first_half, second_half = (
            features[..., :half_dimension],
            features[..., half_dimension:],
        )  # [..., D/2], [..., D/2]
        return torch.cat((-second_half, first_half), dim=-1)  # [..., D]

    def _apply_partial_rotary_embedding(
        self, features: Tensor, sine: Tensor, cosine: Tensor, rotary_dimension: int
    ) -> Tensor:
        """
        Apply rotary embedding to a fraction of the dimension.

        :param Tensor features: tensor of shape [..., D] containing the features to be rotated
        :param Tensor sine: tensor of shape [B, L, R] containing the sine values for the rotary embedding
        :param Tensor cosine: tensor of shape [B, L, R] containing the cosine values for the rotary embedding
        :param int rotary_dimension: Dimension to apply rotary embedding to
        :return Tensor: Tensor of shape [..., D] after applying rotary embedding to the first R dimensions and leaving the rest unchanged
        """

        feature_dtype = features.dtype
        features_rotary = features[..., :rotary_dimension].to(self.dtype)  # [..., R]
        features_passive = features[..., rotary_dimension:]  # [..., D - R]

        # Features have shape [B, ..., L, R], sine and cosine have shape [B, L, R]
        middle_dimensions = features.ndim - sine.ndim
        broadcast_slice = (slice(None),) + (None,) * middle_dimensions + (...,)

        rotated_rotary = (
            (features_rotary * cosine[broadcast_slice])
            + (self._rotate_half(features_rotary) * sine[broadcast_slice])
        ).to(feature_dtype)  # [..., R]
        return torch.cat((rotated_rotary, features_passive), dim=-1)  # [..., D]

    @override
    def forward(self, features: Tensor, positions: Tensor) -> Tensor:
        rotated_features = self._apply_partial_rotary_embedding(
            features,
            *self.trigonometric_position_frequencies(positions),
            self.rotary_dimension,
        )  # [..., D]
        return rotated_features


@final
class CachedRotaryPositionalEncoding(RotaryPositionalEncoding):
    """
    Cached Rotary positional embedding.
    """

    def __init__(
        self,
        dimension: int,
        rotational_fraction: float = 1,
        base: int = 10_000,
        dtype: torch.dtype = torch.float32,
        cache_size: int = 4096,
    ):
        super().__init__(dimension, rotational_fraction, base, dtype)

        frequencies = self.frequencies(
            torch.arange(cache_size, dtype=self.dtype)
        )  # [L, R]

        self.sine_cache = nn.Buffer(frequencies.sin(), persistent=False)
        self.cosine_cache = nn.Buffer(frequencies.cos(), persistent=False)
        self.cache_size = cache_size

    def frequencies(self, positions: Tensor) -> Tensor:
        half_frequencies = torch.einsum(
            "l,d->ld",
            positions,
            self.inverse_frequencies,
        )  # [L, R/2]
        frequencies = torch.cat((half_frequencies, half_frequencies), dim=-1)  # [L, R]
        return frequencies  # [L, R]

    @override
    def trigonometric_position_frequencies(
        self,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.sine_cache[positions], self.cosine_cache[
            positions
        ]  # [B, L, R], [B, L, R]


class ContinuousRotaryPositionalEncoding(RotaryPositionalEncoding):
    """
    Full Rotary positional embedding.
    """

    @override
    def trigonometric_position_frequencies(
        self,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        half_frequencies = torch.einsum(
            "bl,d->bld",
            positions.to(self.dtype),
            self.inverse_frequencies,
        )  # [B, L, R/2]
        frequencies = torch.cat(
            (half_frequencies, half_frequencies), dim=-1
        )  # [B, L, R]
        return frequencies.sin(), frequencies.cos()  # [B, L, R], [B, L, R]
