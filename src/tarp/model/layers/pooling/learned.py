import math
from typing import override

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SelfAttentionPooling(nn.Module):
    """
    Content-dependent single-query attention pooling.

    A learned query vector q ∈ ℝ^D determines token importance dynamically based on feature content.

    Sources:
    "A Structured Self-attentive Sentence Embedding" [(Lin et al., 2017)](https://arxiv.org/abs/1703.03130)
    "Hierarchical Attention Networks for Document Classification" [(Yang et al., 2016)](https://doi.org/10.18653/V1/N16-1174)
    """

    def __init__(self, feature_dimension: int):
        """
        :param int feature_dimension: The dimensionality of the input features.
        """
        super().__init__()
        self.feature_dimension = feature_dimension
        self.query_vector = nn.Parameter(torch.empty(feature_dimension))  # (D,)
        self.reset_parameters()

    def reset_parameters(self):
        _ = nn.init.uniform_(self.query_vector, -0.1, 0.1)

    @override
    def forward(
        self,
        features: Tensor,
        attention_bias: Tensor,
    ) -> Tensor:
        """
        :param Tensor features: Input tensor of shape `(batch_size, sequence_length, feature_dimension)`
        :param Tensor attention_bias: Optional bias tensor of shape `(batch_size, sequence_length)` to be added to the attention scores.
        :param bool return_attention: Whether to return the attention weights.
        :return: The pooled output tensor, and optionally the attention weights.
        :rtype: Union[Tensor, tuple[Tensor, Tensor]]
        """

        # Compute attention scores, scaled by the square root of feature dimension
        scores = torch.einsum("bld,d->bl", features, self.query_vector) / math.sqrt(
            self.feature_dimension
        )  # (B, L)

        scores = scores + attention_bias  # (B, L)

        # Calculate attention weights
        weights = F.softmax(scores, dim=1)  # (B, L)

        # Weighted sum
        context = torch.einsum("bld,bl->bd", features, weights)  # (B, D)
        return context
