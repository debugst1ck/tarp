from torch import Tensor
from transformers import AutoTokenizer

from tarp.services.tokenizers import Tokenizer


class Dnabert2Tokenizer(Tokenizer):
    """
    Wrapper around the DNABERT-2 tokenizer.
    """

    def __init__(self, name: str = "zhihan1996/DNABERT-2-117M"):
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    def tokenize(self, text: str) -> Tensor:
        return self.tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)

    @property
    def pad_token_id(self) -> int:
        return 3

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def mask_token_id(self) -> int:
        return 4
