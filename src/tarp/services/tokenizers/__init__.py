from abc import ABC, abstractmethod
from torch import Tensor

class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> Tensor:
        """
        Tokenizes the input text.

        :param str text: The text to tokenize.
        :return Tensor: A tensor containing the tokenized input.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def mask_token_id(self) -> int:
        raise NotImplementedError