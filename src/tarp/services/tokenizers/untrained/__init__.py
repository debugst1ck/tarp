# Untrained tokenizer implementation
from abc import ABC, abstractmethod
from collections.abc import Iterable

from tarp.services.tokenizers import Tokenizer


class UntrainedTokenizer(Tokenizer, ABC):
    @abstractmethod
    def train(self, data: Iterable[str]) -> None:
        """
        Train the tokenizer on the provided data.

        :param Iterable[str] data: An iterable of strings to train the tokenizer on.
        """
        raise NotImplementedError
