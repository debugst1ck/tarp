from typing import final, override

from torch import Tensor, nn


@final
class GlobalClassificationPooling1D(nn.Module):
    def __init__(self, classification_token_index: int = 0):
        super().__init__()
        self.classification_token_index = classification_token_index

    @override
    def forward(self, features: Tensor, attention_mask: Tensor) -> Tensor:
        """
        :param Tensor features: [B, L, D]
        :param Tensor attention_mask: [B, L], 1 for valid positions, 0 for padding
        :return:
            pooled: [B, D]
        """
        return features[:, self.classification_token_index, :]
