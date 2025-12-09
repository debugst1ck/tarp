from typing import Optional

from torch import Tensor, nn
from transformers import AutoModel

from tarp.model.backbone import Encoder, FrozenModel
from tarp.model.layers.pooling.learned import SelfAttentionPooling


class Dnabert2Encoder(Encoder):
    def __init__(
        self, embedding_dimension: int, name: str = "zhihan1996/DNABERT-2-117M"
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.encoder = AutoModel.from_pretrained(name, trust_remote_code=True)
        self.pooling = SelfAttentionPooling(embedding_dimension)

    def encode(
        self,
        sequence: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_sequence: bool = False,
    ) -> Tensor:
        """
        Encode the input sequence using DNABERT and apply attention pooling.

        :param Tensor sequence: The input sequence for the encoder.
        :param Tensor attention_mask: Optional attention mask for padding tokens. (0 = pad)
        :return: The encoded embeddings.
        :rtype: Tensor
        """
        outputs = self.encoder(input_ids=sequence, attention_mask=attention_mask)[0]
        if return_sequence:
            return outputs
        else:
            pooled_representation = self.pooling(outputs)
            return pooled_representation

    @property
    def encoding_size(self) -> int:
        return self.embedding_dimension


class FrozenDnabert2Encoder(Encoder, FrozenModel):
    def __init__(self, hidden_dimension: int, name: str = "zhihan1996/DNABERT-2-117M"):
        super().__init__()
        self.hidden_dimension = hidden_dimension
        self.encoder: nn.Module = AutoModel.from_pretrained(
            name, trust_remote_code=True
        )
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
        """
        Encode the input sequence using a frozen DNABERT and apply attention pooling.

        :param Tensor sequence: The input sequence for the encoder.
        :param Tensor attention_mask: Optional attention mask for padding tokens. (0 = pad)
        :return: The encoded embeddings.
        :rtype: Tensor
        """
        outputs = self.encoder(input_ids=sequence, attention_mask=attention_mask)[0]
        if return_sequence:
            return outputs
        else:
            pooled_representation = self.pooling(outputs)
            return pooled_representation
