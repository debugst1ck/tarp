# Character level
import torch
from torch import Tensor

from tarp.services.tokenizers import Tokenizer


class CharacterTokenizer(Tokenizer):
    @property
    def pad_token_id(self) -> int:
        return 0x00  # NUL character

    @property
    def mask_token_id(self) -> int:
        return 0x1A  # SUB character

    @property
    def vocab_size(self) -> int:
        return 256  # ASCII

    def tokenize(self, text: str) -> Tensor:
        return torch.frombuffer(
            bytearray(text.encode("ascii", "replace")), dtype=torch.uint8
        )
