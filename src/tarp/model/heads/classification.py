from typing import override

from torch import Tensor, nn

from tarp.model.backbone.core import Encoder


class ClassificationModel(nn.Module):
    def __init__(
        self, embedding: nn.Embedding, encoder: Encoder, number_of_classes: int
    ):
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
        payload_mask: Tensor | None = None,
        positions: Tensor | None = None,
    ) -> Tensor:
        sequence_embeddings = self.embedding(sequence)
        pooled_output = self.encoder(
            sequence_embeddings,
            attention_mask,
            payload_mask=payload_mask,
            positions=positions,
            mode="pooled",
        )
        return self.classification_head(pooled_output)
