# CLS pooling layer for BERT-like models
#
from typing import Optional, Union

import torch
from torch import Tensor, nn


class CLSPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """
        :param input: Tensor of shape (batch_size, sequence_length, feature_dimension)
        :param attention_mask: Optional attention mask of shape (batch_size, sequence_length)
        :param return_attention: Whether to return attention weights.
        :return: Pooled tensor of shape (batch_size, feature_dimension), optionally with attention weights.
        """
        # Take the first token's embedding
        pooled = input[:, 0, :]  # shape: (batch_size, feature_dimension)

        if return_attention:
            if attention_mask is not None:
                attention_weights = attention_mask.float() / attention_mask.sum(
                    dim=1, keepdim=True
                )
            else:
                attention_weights = torch.full(
                    (input.size(0), input.size(1)),
                    1.0 / input.size(1),
                    device=input.device,
                )
            return pooled, attention_weights

        return pooled
