# Untrained tokenizer implementation
from abc import ABC, abstractmethod
from typing import Sequence

from tarp.services.tokenizers import Tokenizer


class UntrainedTokenizer(Tokenizer, ABC):
    @abstractmethod
    def train(
        self, texts: Sequence[str], vocabulary_size: int = 1024, **kwargs
    ) -> None:
        """
        Optional: Train tokenizer on provided texts.
        Only implemented for trainable subclasses.

        :param Sequence[str] texts: List of texts to train the tokenizer on.
        :param int vocabulary_size: Desired vocabulary size.
        :param kwargs: Additional parameters for training.
        :return: None
        """
        raise NotImplementedError
