from typing import final, override

from torch import Tensor
from transformers import AutoTokenizer

from tarp.preprocessing.tokenizers.core import Tokenizer


@final
class DnaBert2Tokenizer(Tokenizer):
    def __init__(self, name: str = "zhihan1996/DNABERT-2-117M"):
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    @override
    def encode(self, text: str) -> Tensor:
        return self.tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)

    @override
    def decode(self, tokens: Tensor) -> str:
        return self.tokenizer.decode(tokens.tolist(), skip_special_tokens=True)

    @property
    @override
    def vocabulary_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    @override
    def pad_token_id(self) -> int:
        return 3

    @property
    @override
    def mask_token_id(self) -> int:
        return 4

    @property
    @override
    def cls_token_id(self) -> int:
        return 1
