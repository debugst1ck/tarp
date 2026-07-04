from abc import ABC, abstractmethod
from typing import final, override

from torch import Tensor, nn


class TransformativePositionalEncoding(nn.Module, ABC):
    """
    Interface for positional encodings that directly mutate individual feature vectors (e.g., Absolute Embeddings, RoPE).
    """

    @override
    @abstractmethod
    def forward(self, features: Tensor, positions: Tensor) -> Tensor:
        """
        :param Tensor features: Features tensor of shape [B, ..., L, D]
        :param Tensor positions: Position indices of shape [B, L]
        :return Tensor: The modified/encoded features of shape [B, ..., L, D]
        """
        raise NotImplementedError


class AttentionBiasPositionalEncoding(nn.Module, ABC):
    """
    Interface for positional encodings that produce an attention bias tensor (e.g., ALiBi, T5 Relative Bias).
    """

    @override
    @abstractmethod
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        *,
        query_positions: Tensor,
        key_positions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """
        :param Tensor query: Query tensor of shape [B, ..., L, D]
        :param Tensor key: Key tensor of shape [B, ..., L, D]
        :param Tensor query_positions: Position indices for the query tokens, shape [B, L]
        :param Tensor key_positions: Position indices for the key tokens, shape [B, L]
        :return tuple[Tensor, Tensor, Tensor | None]: (query, key, attention_bias) optional attention bias tensor of shape [B, 1, L, L] or [B, 1, 1, L]
        """
        return query, key, None  # Placeholder for attention bias


class HeterogeneousTransformativePositionalEncoding(AttentionBiasPositionalEncoding):
    def __init__(
        self,
        query_encoder: TransformativePositionalEncoding,
        key_encoder: TransformativePositionalEncoding,
    ):
        super().__init__()
        self.query_encoder = query_encoder
        self.key_encoder = key_encoder

    @override
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        *,
        query_positions: Tensor,
        key_positions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        encoded_query = self.query_encoder(query, query_positions)
        encoded_key = self.key_encoder(key, key_positions)
        return encoded_query, encoded_key, None


@final
class NoPositionalEncoding(AttentionBiasPositionalEncoding):
    def __init__(self):
        super().__init__()

    @override
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        *,
        query_positions: Tensor,
        key_positions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        return query, key, None
