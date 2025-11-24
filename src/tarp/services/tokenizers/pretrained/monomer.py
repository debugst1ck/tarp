# Nucleotide tokenizer no pretraining
import torch
from torch import Tensor

from tarp.services.tokenizers import Tokenizer


class NucleotideTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        self.vocab = {
            "A": 0,  # Adenine
            "C": 1,  # Cytosine
            "G": 2,  # Guanine
            "T": 3,  # Thymine
            "N": 4,  # Unknown nucleotide also <UNK>
            "<PAD>": 5,
            "<MASK>": 6,
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        self.lookup = torch.full((256,), self.vocab["N"], dtype=torch.long)

        for char, idx in self.vocab.items():
            if len(char) == 1:
                self.lookup[ord(char)] = idx

    def tokenize(self, text: str) -> Tensor:
        bytes = text.encode("ascii", errors="ignore")
        indices = self.lookup[
            torch.frombuffer(bytearray(bytes), dtype=torch.uint8).long()
        ]
        return indices

    @property
    def pad_token_id(self) -> int:
        return self.vocab["<PAD>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def mask_token_id(self) -> int:
        return self.vocab["<MASK>"]


class ProteinTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        amino_acids = [
            "A",  # Alanine
            "R",  # Arginine
            "N",  # Asparagine
            "D",  # Aspartic acid
            "C",  # Cysteine
            "Q",  # Glutamine
            "E",  # Glutamic acid
            "G",  # Glycine
            "H",  # Histidine
            "I",  # Isoleucine
            "L",  # Leucine
            "K",  # Lysine
            "M",  # Methionine
            "F",  # Phenylalanine
            "P",  # Proline
            "S",  # Serine
            "T",  # Threonine
            "W",  # Tryptophan
            "Y",  # Tyrosine
            "V",  # Valine
            "X",  # Unknown amino acid
            "<PAD>",
            "<MASK>",
        ]
        self.vocab = {aa: idx for idx, aa in enumerate(amino_acids)}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        self.lookup = torch.full((256,), self.vocab["X"], dtype=torch.long)

        for char, idx in self.vocab.items():
            if len(char) == 1:
                self.lookup[ord(char)] = idx

    def tokenize(self, text: str) -> Tensor:
        indices = self.lookup[
            torch.frombuffer(
                bytearray(text.encode("ascii", errors="ignore")),
                dtype=torch.uint8,
            ).long()
        ]
        return indices

    @property
    def pad_token_id(self) -> int:
        return self.vocab["<PAD>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def mask_token_id(self) -> int:
        return self.vocab["<MASK>"]
