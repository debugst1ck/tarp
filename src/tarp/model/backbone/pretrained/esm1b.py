from typing import Optional

from torch import Tensor, nn
from transformers import AutoModel

from tarp.model.backbone import Encoder, FrozenModel
from tarp.model.layers.pooling.learned import SelfAttentionPooling


class Esm1bEncoder(Encoder):
    def __init__(self, model_name="facebook/esm1b_t33_650M_UR50S"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        self.pooling = SelfAttentionPooling(self.hidden_size)

    def encode(
        self,
        sequence: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_sequence=False,
    ) -> Tensor:
        outputs = self.model(
            input_ids=sequence,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state[:, 1:, :]  # Remove BOS token
        if return_sequence:
            return hidden
        else:
            pooled_representation = self.pooling(
                hidden,
                attention_mask=attention_mask[:, 1:]
                if attention_mask is not None
                else None,
            )
            return pooled_representation

    @property
    def encoding_size(self):
        return self.hidden_size


class FrozenEsm1bEncoder(Encoder, FrozenModel):
    def __init__(
        self, hidden_dimension: int, model_name="facebook/esm1b_t33_650M_UR50S"
    ):
        super().__init__()
        self.hidden_dimension = hidden_dimension
        self.encoder: nn.Module = AutoModel.from_pretrained(model_name)
        self.pooling = SelfAttentionPooling(hidden_dimension)

        # Freeze the encoder parameters
        self.freeze()

    @property
    def encoding_size(self) -> int:
        return self.hidden_dimension

    def freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def unfreeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.encoder.train()

    def encode(
        self,
        sequence: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_sequence: bool = False,
    ) -> Tensor:
        outputs = self.encoder(
            input_ids=sequence,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state[:, 1:, :]  # Remove BOS token
        if return_sequence:
            return hidden
        else:
            pooled_representation = self.pooling(
                hidden,
                attention_mask=attention_mask[:, 1:]
                if attention_mask is not None
                else None,
            )
            return pooled_representation
