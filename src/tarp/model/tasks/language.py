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

        # Tied embeddings are highly anisotropic (0.078 vs 0.996 for untied inputs).
        # Untied models learn near-orthogonal input and output representations.

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
