from typing import final, override

from torch import Tensor, nn


@final
class GlobalAveragePooling1D(nn.Module):
    @override
    def forward(self, features: Tensor, attention_mask: Tensor) -> Tensor:
        """
        :param Tensor features: [B, L, D]
        :param Tensor attention_mask: [B, L], 1 for valid positions, 0 for padding
        :return:
            pooled: [B, D]
        """
        masked_features = features * attention_mask.unsqueeze(-1).to(
            dtype=features.dtype
        )  # [B, L, D]
        pooling_accumulation = masked_features.sum(dim=1)  # [B, D]
        length = attention_mask.sum(dim=1).unsqueeze(-1).clamp(min=1)  # [B, 1]
        return pooling_accumulation / length  # [B, D]


@final
class GlobalMaximumPooling1D(nn.Module):
    @override
    def forward(self, features: Tensor, attention_mask: Tensor) -> Tensor:
        """
        :param Tensor features: [B, L, D]
        :param Tensor attention_mask: [B, L], 1 for valid positions, 0 for padding
        :return:
            pooled: [B, D]
        """
        masked_features = features * attention_mask.unsqueeze(-1).to(
            dtype=features.dtype
        )  # [B, L, D]
        return masked_features.max(dim=1).values  # [B, D]
