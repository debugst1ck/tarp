# Mamba implementation for Encoder
from typing import Optional

from mambapy.mamba import Mamba, MambaConfig
from torch import Tensor, nn

from tarp.model.backbone import Encoder
from tarp.model.layers.pooling.learned import SelfAttentionPooling


class MambaEncoder(Encoder):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int,
        number_of_layers: int = 6,
        padding_id: int = 0,
    ) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension
        self.number_of_layers = number_of_layers
        self.padding_id = padding_id

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dimension,
            padding_idx=padding_id,
        )

        config = MambaConfig(
            d_model=embedding_dimension,
            n_layers=number_of_layers,
        )

        self.mamba = Mamba(config)
        self.pooling = SelfAttentionPooling(
            self.embedding_dimension,
        )

    def encode(
        self,
        sequence: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_sequence: bool = False,
    ) -> Tensor:
        # Mamba doesn't support attention masks yet, so we have to pack the sequences manually, then repad them

        # Embed the input tokensinput
        embeddings = self.embedding(sequence)  # (B, L, D)

        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)
        # This is a duck tape maybe fix cause mamba can still propagate padding info in layers

        output = self.mamba(embeddings)  # (B, L, D)

        if attention_mask is not None:
            # Set padding positions to zero, broadcast over embedding dimension
            output = output * attention_mask.unsqueeze(-1)
        # Does not prevent the model from attending to or learning from those positions
        # But the output representations for those positions will be zeroed out

        if return_sequence:
            return output  # (B, L, D)
        else:
            pooled_output = self.pooling(output, attention_mask)  # (B, D)
            return pooled_output

    @property
    def encoding_size(self) -> int:
        return self.embedding_dimension
