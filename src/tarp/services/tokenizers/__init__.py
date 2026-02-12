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
    def vocab_size(self) -> int:
        return None

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        return None

    @property
    @abstractmethod
    def mask_token_id(self) -> int:
        return None

    @property
    @abstractmethod
    def cls_token_id(self) -> int:
        return None

    @property
    def special_tokens_and_ids(self) -> dict[str, int]:
        """
        Returns a dictionary of special token names and their corresponding token IDs.

        :return dict[str, int]: A dictionary mapping special token names to their token IDs.
        """
        return {
            name: token_id
            for name, token_id in (
                ("pad", self.pad_token_id),
                ("mask", self.mask_token_id),
                ("cls", self.cls_token_id),
            )
            if token_id is not None
        }
