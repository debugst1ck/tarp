from abc import ABC, abstractmethod
from typing import override

import torch
from torch import Tensor

from tarp.model.layers.positional.core import (
    TransformativePositionalEncoding,
)


class RotaryPositionalEncoding(TransformativePositionalEncoding, ABC):
    def __init__(
        self,
        dimension: int,
        rotational_fraction: float = 1.0,
        base: int = 10000,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.dimension = dimension  # [D]
        self.rotational_fraction = rotational_fraction
        self.base = base
        self.rotary_dimension = int(dimension * rotational_fraction)  # [R]

        # The dtype for core calculations
        self.dtype = dtype

        self.inverse_frequencies: Tensor
        self.register_buffer(
            "inverse_frequencies",
            torch.empty(self.rotary_dimension // 2),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        inverse_frequencies = 1.0 / (
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
        )  # [R/2]

        with torch.no_grad():
            _ = self.inverse_frequencies.copy_(inverse_frequencies)

    @abstractmethod
    def trigonometric_position_frequencies(
        self, positions: Tensor, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the sine and cosine components for the given positions.

        :param Tensor positions: [B, L] The positions for which to compute the rotations.
        :param torch.device device: The device on which to compute the rotations.
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
            *self.trigonometric_position_frequencies(positions, features.device),
            self.rotary_dimension,
        )  # [..., D]
        return rotated_features


class CachedIntegerRotaryPositionalEncoding(RotaryPositionalEncoding):
    def __init__(
        self,
        dimension: int,
        rotational_fraction: float = 1.0,
        base: int = 10000,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            dimension=dimension,
            rotational_fraction=rotational_fraction,
            base=base,
            dtype=dtype,
        )
        self.trigonometric_cache = torch.empty(0, device="cpu", dtype=dtype)
        self.cache_length: int = 0

    def _trigonometric_cache(self, length: int, device: torch.device) -> Tensor:
        if self.cache_length >= length and self.trigonometric_cache.device == device:
            return self.trigonometric_cache

        allocation_length = max(length, self.cache_length * 2)
        positions = torch.arange(
            allocation_length, device=device, dtype=self.dtype
        )  # [L]
        half_frequencies = torch.outer(positions, self.inverse_frequencies)  # [L, R/2]
        frequencies = torch.cat((half_frequencies, half_frequencies), dim=-1)

        self.trigonometric_cache = torch.stack(
            (frequencies.sin(), frequencies.cos()), dim=-1
        )  # [L, R, 2]
        self.cache_length = allocation_length
        return self.trigonometric_cache

    @override
    def trigonometric_position_frequencies(
        self, positions: Tensor, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        cache = self._trigonometric_cache(
            int(positions.max().item() + 1), device
        )  # [L, R, 2]
        valid = cache[positions]  # [B, L, R, 2]
        return valid[..., 0], valid[..., 1]  # [B, L, R], [B, L, R]


class ContinuousRotaryPositionalEncoding(RotaryPositionalEncoding):
    """
    Rotary positional embedding with fractional position support.
    """

    @override
    def trigonometric_position_frequencies(
        self, positions: Tensor, device: torch.device
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
