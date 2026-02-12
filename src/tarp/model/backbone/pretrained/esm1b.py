from typing import Optional

from torch import Tensor
from transformers import AutoModel

from tarp.model.backbone import Encoder, FrozenModel
from tarp.model.layers.pooling.reductions import GlobalMeanPooling


class Esm1bEncoder(Encoder):
    def __init__(self, name="facebook/esm1b_t33_650M_UR50S"):
        super().__init__()
        self.model = AutoModel.from_pretrained(name)
        self.model_dimension = self.model.config.hidden_size
        self.pooling = GlobalMeanPooling()

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
        return self.model_dimension


class FrozenEsm1bEncoder(Encoder, FrozenModel):
    def __init__(self, name="facebook/esm1b_t33_650M_UR50S"):
        super().__init__()
        self.model = AutoModel.from_pretrained(name)
        self.model_dimension = self.model.config.hidden_size
        self.pooling = GlobalMeanPooling()

        # Freeze the encoder parameters
        self.freeze()

    @property
    def encoding_size(self) -> int:
        return self.model_dimension

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()

    def encode(
        self,
        sequence: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_sequence: bool = False,
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
