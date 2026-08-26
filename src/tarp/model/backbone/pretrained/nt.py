from typing import Literal, overload, override

import torch
import transformers.pytorch_utils
from torch import Tensor
from transformers import AutoConfig, AutoModelForMaskedLM, PreTrainedModel

from tarp.model.backbone.core import Encoder
from tarp.model.layers.pooling.atomic import GlobalAveragePooling1D

# Some magical voodoo to make the Esm NTv2 model work with older versions of transformers.
# The Esm NTv2 model relies on a function that was added in transformers v4.30.0, so we need to add it if it's not present.
if not hasattr(transformers.pytorch_utils, "find_pruneable_heads_and_indices"):

    def find_pruneable_heads_and_indices(
        heads, n_heads, head_size, already_pruned_heads
    ):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            mask[head] = 0
        mask = mask.reshape(-1).eq(1)
        index = torch.arange(len(mask))[mask]
        return heads, index

    transformers.pytorch_utils.find_pruneable_heads_and_indices = (
        find_pruneable_heads_and_indices
    )


# More magical voodoo to make the Esm NTv2 model work with older versions of transformers.
def _get_all_tied_weights_keys(self):
    tied = getattr(self, "_tied_weights_keys", None)
    if tied is None:
        return {}
    if isinstance(tied, dict):
        return tied
    if isinstance(tied, (list, tuple, set)):
        return {k: k for k in tied}
    return tied


def _set_all_tied_weights_keys(self, value):
    self._tied_weights_keys = value


# Thank god for python's dynamic attributes
PreTrainedModel.all_tied_weights_keys = property(
    _get_all_tied_weights_keys, _set_all_tied_weights_keys
)


class NucleotideTransformerV2Encoder(Encoder):
    def __init__(
        self, name: str = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(name, trust_remote_code=True)

        # Inject attributes because Esm NTv2 rely on them being present in the config object
        config.is_decoder = False
        config.add_cross_attention = False

        self.model = AutoModelForMaskedLM.from_pretrained(
            name, config=config, trust_remote_code=True
        )

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
        )
        hidden = outputs["hidden_states"][-1]  # [B, L, D]
        pooled = self.pooling(hidden, attention_mask)  # [B, D]
        match mode:
            case "sequence":
                return hidden, None
            case "pooled":
                return pooled, None
            case "both":
                return hidden, pooled, None
