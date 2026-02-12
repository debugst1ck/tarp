# Metric Learning Finetuning Module
from typing import Optional

from torch import Tensor, nn

from tarp.model.backbone import Encoder


class TripletMetricModel(nn.Module):
    def __init__(self, encoder: Encoder):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        anchor: Tensor,
        positive: Tensor,
        negative: Tensor,
        anchor_mask: Optional[Tensor] = None,
        positive_mask: Optional[Tensor] = None,
        negative_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        :param Tensor anchor: The anchor sequences.
        :param Tensor positive: The positive sequences.
        :param Tensor negative: The negative sequences.
        :param Optional[Tensor] anchor_mask: Optional attention mask for padding tokens in the anchor. (0 = pad)
        :param Optional[Tensor] positive_mask: Optional attention mask for padding tokens in the positive. (0 = pad)
        :param Optional[Tensor] negative_mask: Optional attention mask for padding tokens in the negative. (0 = pad)
        :return: The encoded representations of the anchor, positive, and negative sequences.
        :rtype: tuple[Tensor, Tensor, Tensor]
        """
        anchor_representation = self.encoder(anchor, anchor_mask)
        positive_representation = self.encoder(positive, positive_mask)
        negative_representation = self.encoder(negative, negative_mask)

        return anchor_representation, positive_representation, negative_representation
