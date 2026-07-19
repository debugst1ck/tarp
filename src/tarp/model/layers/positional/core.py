from abc import ABC, abstractmethod
from typing import override

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


class NoPositionalEncoding(TransformativePositionalEncoding):
    """
    A no-op positional encoding that returns the input features unchanged.
    """

    @override
    def forward(self, features: Tensor, positions: Tensor) -> Tensor:
        return features
