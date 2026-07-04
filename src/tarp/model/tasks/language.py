from typing import final, override

from torch import Tensor, nn

from tarp.cli.core import Console
from tarp.model.backbone.core import Encoder


@final
class LanguageModel(nn.Module):
    def __init__(
        self,
        embedding: nn.Module,
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
        if (
            isinstance(self.embedding, nn.Embedding)
            and self.embedding.weight.shape == self.language_head.weight.shape
        ):
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
        sequence_embeddings = self.embedding(sequence)  # [B, L, D]
        encoded, auxiliary_loss = self.encoder(
            sequence_embeddings,
            attention_mask,
            positions=positions,
            mode="sequence",
        )  # [B, L, D]
        return self.language_head(encoded), auxiliary_loss  # [B, L, V]
