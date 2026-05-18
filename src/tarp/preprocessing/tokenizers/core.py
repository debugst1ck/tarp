from abc import ABC, abstractmethod
from typing import Final

from torch import Tensor


class Tokenizer(ABC):
    MISSING_ID: Final[int] = -1

    @abstractmethod
    def encode(self, text: str) -> Tensor:
        """
        Encodes the input text into tokens.

        :param str text: The input text to encode.
        :return Tensor: A tensor containing the token IDs corresponding to the input text.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, tokens: Tensor) -> str:
        """
        Decodes the input tokens back into text.

        :param Tensor tokens: A tensor containing token IDs to decode.
        :return str: The decoded text corresponding to the input tokens.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def vocabulary_size(self) -> int:
        """
        :return int: The size of the vocabulary used.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        return self.MISSING_ID

    @property
    @abstractmethod
    def mask_token_id(self) -> int:
        return self.MISSING_ID

    @property
    @abstractmethod
    def cls_token_id(self) -> int:
        return self.MISSING_ID

    @property
    def special_tokens_and_ids(self) -> dict[str, int]:
        """
        :return dict[str, int]: A dictionary mapping special token names (e.g., "pad", "mask", "cls") to their corresponding token IDs. Only includes tokens that are defined (i.e., not None).
        """
        specials: dict[str, int] = {
            "pad": self.pad_token_id,
            "mask": self.mask_token_id,
            "cls": self.cls_token_id,
        }
        return {
            name: tid for name, tid in specials.items() if tid is not self.MISSING_ID
        }
