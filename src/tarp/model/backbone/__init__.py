from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor, nn


class Encoder(nn.Module, ABC):
    """
    Abstract base class for all encoder models.
    An encoder model takes a sequence of embeddings as input and produces
    either a sequence of encoded embeddings or a single encoded embedding.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def encode(
        self,
        sequence: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_sequence: bool = False,
    ) -> Tensor:
        """
        Encodes a sequence of embeddings.

        :param Tensor sequence: A tensor of shape `(batch_size, sequence_length, embedding_dimension)` representing the input sequence of embeddings.
        :param Tensor attention_mask: An optional tensor of shape `(batch_size, sequence_length)` representing the attention mask for the input sequence.
        :param Tensor return_sequence: A boolean indicating whether to return the full sequence of encoded embeddings or a single pooled embedding.
        :return Tensor: A tensor of shape `(batch_size, encoding_size)` if return_sequence is `False`, or `(batch_size, sequence_length, encoding_size)` if `True`.
        """
        pass

    @property
    @abstractmethod
    def encoding_size(self) -> int:
        pass


class FrozenModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def freeze(self):
        pass

    @abstractmethod
    def unfreeze(self):
        pass
