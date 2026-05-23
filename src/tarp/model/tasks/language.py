from typing import override

from torch import Tensor, nn

from tarp.cli.core import Console
from tarp.model.backbone.core import Encoder


class LanguageModel(nn.Module):
    def __init__(
        self,
        embedding: nn.Embedding,
        encoder: Encoder,
        vocabulary_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.embedding = embedding
        self.language_head = nn.Linear(
            self.encoder.encoding_size,
            vocabulary_size,
            bias=bias,
        )
        if self.embedding.weight.shape == self.language_head.weight.shape:
            self.language_head.weight = self.embedding.weight
            Console.warning("Tied language head weights to encoder embedding weights.")

    @override
    def forward(
        self,
        sequence: Tensor,
        attention_mask: Tensor,
        *,
        positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        sequence_embeddings = self.embedding(sequence)
        encoded, auxillary_loss = self.encoder(
            sequence_embeddings,
            attention_mask,
            positions=positions,
            mode="sequence",
        )
        return self.language_head(encoded), auxillary_loss
