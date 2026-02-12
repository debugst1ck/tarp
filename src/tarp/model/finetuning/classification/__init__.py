import torch
from torch import nn

from tarp.model.backbone import Encoder


class ClassificationModel(nn.Module):
    """
    A simple classification model.
    """

    def __init__(self, encoder: Encoder, number_of_classes: int):
        super().__init__()
        self.number_of_classes = number_of_classes
        self.encoder = encoder
        self.classification_head = nn.Linear(
            self.encoder.encoding_size, number_of_classes
        )

    def forward(
        self,
        sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        :param Tensor sequence: The input sequence for the encoder.
        :param Tensor attention_mask: Optional attention mask for padding tokens. (0 = pad)
        :return: The classification logits.
        :rtype: Tensor
        """
        pooled_representation = self.encoder(
            sequence, attention_mask, return_sequence=return_sequence
        )
        return self.classification_head(pooled_representation)
