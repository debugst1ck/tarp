import torch
from torch import nn

from tarp.cli.logging import Console
from tarp.model.backbone import Encoder


class LanguageModel(nn.Module):
    """
    A simple language modeling model.
    """

    def __init__(self, encoder: Encoder, vocabulary_size: int):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.encoder = encoder
        self.language_head = nn.Linear(
            self.encoder.encoding_size, vocabulary_size, bias=False
        )
        # Tie weights between language head and input embeddings
        # source: https://arxiv.org/abs/1608.05859
        if (
            hasattr(self.encoder, "embedding")
            and isinstance(self.encoder.embedding, nn.Embedding)
            and self.encoder.embedding.weight.shape == self.language_head.weight.shape
        ):
            Console.warning(
                "Tying weights between encoder embeddings and language head"
            )
            # Disable language head bias if weights are tied
            self.language_head = nn.Linear(
                self.encoder.encoding_size, vocabulary_size, bias=False
            )
            # Tie the weights
            self.language_head.weight = self.encoder.embedding.weight

    @property
    def tied(self) -> bool:
        """
        :return: Whether the language head weights are tied to the encoder embeddings.
        :rtype: bool
        """
        if (
            hasattr(self.encoder, "embedding")
            and isinstance(self.encoder.embedding, nn.Embedding)
            and self.encoder.embedding.weight.shape == self.language_head.weight.shape
        ):
            return self.language_head.weight is self.encoder.embedding.weight
        return False

    def forward(
        self,
        sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        return_sequence: bool = True,
    ) -> torch.Tensor:
        """
        :param Tensor sequence: The input sequence for the encoder.
        :param Tensor attention_mask: Optional attention mask for padding tokens. (0 = pad)
        :return: The language modeling logits.
        :rtype: Tensor
        """
        encoded_representation = self.encoder(
            sequence, attention_mask, return_sequence=return_sequence
        )
        return self.language_head(encoded_representation)
