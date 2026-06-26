from typing import final, override

from torch import Tensor, nn

from tarp.model.backbone.core import Encoder


@final
class ClassificationModel(nn.Module):
    def __init__(self, embedding: nn.Module, encoder: Encoder, number_of_classes: int):
        super().__init__()
        self.encoder = encoder
        self.embedding = embedding
        self.classification_head = nn.Linear(
            self.encoder.encoding_size,
            number_of_classes,
        )

    @override
    def forward(
        self,
        sequence: Tensor,
        attention_mask: Tensor,
        *,
        positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        sequence_embeddings = self.embedding(sequence)  # [B, L, D]
        pooled, auxiliary_loss = self.encoder(
            sequence_embeddings,
            attention_mask,
            positions=positions,
            mode="pooled",
        )  # [B, D]
        return self.classification_head(pooled), auxiliary_loss  # [B, C]
