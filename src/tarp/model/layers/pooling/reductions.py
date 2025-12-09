from typing import Optional, Union

import torch
from torch import Tensor, nn


class GlobalMeanPooling(nn.Module):
    """
    Global mean pooling layer.
    Computes the mean of embeddings along the sequence dimension,
    optionally ignoring padded tokens using an attention mask.
    """

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
        :param attention_mask: Optional attention mask of shape (batch_size, sequence_length) with 1 for valid.
        :param return_attention: Whether to return the attention weights.
        :return: Pooled tensor of shape (batch_size, feature_dimension), optionally with attention weights.
        """
        if attention_mask is not None:
            # Make sure mask is float
            mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
            masked_input = input * mask
            sum_embeddings = masked_input.sum(dim=1)  # sum over sequence
            lengths = mask.sum(dim=1)  # number of valid tokens
            pooled = sum_embeddings / torch.clamp(lengths, min=1e-9)  # avoid div by 0
        else:
            pooled = input.mean(dim=1)

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
