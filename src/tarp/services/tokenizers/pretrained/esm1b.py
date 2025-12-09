from torch import Tensor
from transformers import AutoTokenizer

from tarp.services.tokenizers import Tokenizer


class Esm1bTokenizer(Tokenizer):
    def __init__(self, name="facebook/esm1b_t33_650M_UR50S"):
        self.tokenizer = AutoTokenizer.from_pretrained(name)

    def tokenize(self, text: str) -> Tensor:
        return self.tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def mask_token_id(self):
        return self.tokenizer.mask_token_id
