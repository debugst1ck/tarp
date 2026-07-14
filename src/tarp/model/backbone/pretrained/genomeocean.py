from typing import Literal, overload, override

from torch import Tensor
from transformers import AutoModel

from tarp.model.backbone.core import Encoder
from tarp.model.layers.pooling.atomic import GlobalAveragePooling1D


class GenomeOceanEncoder(Encoder):
    def __init__(self, name: str = "DOEJGI/GenomeOcean-500M"):
        super().__init__()
        self.model = AutoModel.from_pretrained(name)
        self.pooling = GlobalAveragePooling1D()
        self.model_dimension = self.model.config.hidden_size

    @property
    @override
    def encoding_size(self) -> int:
        return self.model_dimension

    @overload
    def encode(
        self,
        sequence_embeddings: Tensor,
        attention_mask: Tensor,
        *,
        positions: Tensor | None = None,
        mode: Literal["sequence", "pooled"],
    ) -> tuple[Tensor, Tensor | None]: ...
    @overload
    def encode(
        self,
        sequence_embeddings: Tensor,
        attention_mask: Tensor,
        *,
        positions: Tensor | None = None,
        mode: Literal["both"],
    ) -> tuple[Tensor, Tensor, Tensor | None]: ...
    @override
    def encode(
        self,
        sequence_embeddings: Tensor,
        attention_mask: Tensor,
        *,
        positions: Tensor | None = None,
        mode: Literal["sequence", "pooled", "both"],
    ) -> tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor, Tensor | None]:
        """
        :param Tensor sequence_embeddings: [B, L], token ids for each sequence
        :param Tensor attention_mask: [B, L], 1 for valid positions, 0 for padding
        :param Tensor positions: [B, L], absolute positions for each token, optional
        :param str mode: "sequence" to return only sequence features [B, L, D], "pooled" to return only pooled features [B, D], "both" to return both
        :return tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor, Tensor | None]:
            If mode is "sequence": (sequence_features, auxiliary_loss)
            If mode is "pooled": (pooled_features, auxiliary_loss)
            If mode is "both": (sequence_features, pooled_features, auxiliary_loss)
        """
        outputs = self.model(
            input_ids=sequence_embeddings,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state  # [B, L, D]
        pooled = self.pooling(hidden, attention_mask)  # [B, D]
        match mode:
            case "sequence":
                return hidden, None
            case "pooled":
                return pooled, None
            case "both":
                return hidden, pooled, None
