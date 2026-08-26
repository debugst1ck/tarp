from typing import final, override

from torch import Tensor
from transformers import AutoTokenizer

from tarp.preprocessing.tokenizers.core import Tokenizer


@final
class NucleotideTransformerV2Tokenizer(Tokenizer):
    def __init__(
        self, name: str = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            name, trust_remote_code=True, padding_side="right"
        )

    @override
    def encode(self, text: str) -> Tensor:
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].squeeze(0)

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
        return self.tokenizer.pad_token_id

    @property
    @override
    def mask_token_id(self) -> int:
        return self.tokenizer.mask_token_id

    @property
    @override
    def cls_token_id(self) -> int:
        return self.tokenizer.cls_token_id
