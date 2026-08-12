from typing import Final, final, override

import torch
from torch import Tensor

from tarp.preprocessing.tokenizers.core import Tokenizer


@final
class NucleotideTokenizer(Tokenizer):
    """
    For ATGCN with special tokens.
    """

    def __init__(self):
        self.vocabulary: Final[dict[str, int]] = {
            "A": 0,  # Adenine
            "C": 1,  # Cytosine
            "G": 2,  # Guanine
            "T": 3,  # Thymine
            "N": 4,  # Unknown nucleotide also <UNK>
            "<PAD>": 5,
            "<MASK>": 6,
            "<CLS>": 7,
        }
        self.inverse_vocabulary: Final[dict[int, str]] = {
            v: k for k, v in self.vocabulary.items()
        }

        self.lookup = torch.full((256,), self.vocabulary["N"], dtype=torch.long)
        for character, index in self.vocabulary.items():
            if len(character) == 1:
                self.lookup[ord(character.upper())] = index
                self.lookup[ord(character.lower())] = index

    @override
    def encode(self, text: str) -> Tensor:
        indices = self.lookup[
            torch.frombuffer(
                bytearray(text.encode("ascii", errors="ignore")), dtype=torch.uint8
            ).long()
        ]
        return indices

    @override
    def decode(self, tokens: Tensor) -> str:
        return "".join(
            [self.inverse_vocabulary.get(int(token.item()), "N") for token in tokens]
        )

    @property
    @override
    def vocabulary_size(self) -> int:
        return len(self.vocabulary)

    @property
    @override
    def pad_token_id(self) -> int:
        return self.vocabulary["<PAD>"]

    @property
    @override
    def mask_token_id(self) -> int:
        return self.vocabulary["<MASK>"]

    @property
    @override
    def cls_token_id(self) -> int:
        return self.vocabulary["<CLS>"]
