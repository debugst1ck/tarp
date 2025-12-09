import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LearnedPositionPooling(nn.Module):
    """
    Learnable position-weighted pooling layer.

    This module pools a sequence of embeddings into a single vector
    using a learnable scalar weight for each absolute position.

    This pooling does NOT depend on token content but only token position.

    Sources:
    "Attention Is All You Need" [(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
    "Bag of Tricks for Efficient Text Classification" [(Joulin et al., 2017)](https://arxiv.org/abs/1607.01759)
    """

    def __init__(self, maximum_sequence_length: int):
        """
        :param int maximum_sequence_length: The maximum length of input sequences.
        """
        super().__init__()
        self.maximum_sequence_length = maximum_sequence_length
        self.position_weights = nn.Parameter(torch.empty(maximum_sequence_length))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.position_weights, -0.1, 0.1)

    def forward(
        self,
        input: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """
        :param Tensor input: Input tensor of shape `(batch_size, sequence_length, feature_dimension)`
        :param Tensor attention_mask: Optional attention mask for padding tokens. (0 = pad)
        :param bool return_attention: Whether to return the attention weights.
        :return: The pooled output tensor, and optionally the attention weights.
        :rtype: Union[Tensor, tuple[Tensor, Tensor]]
        """
        batch_size, sequence_length, _ = input.shape

        # Use only the first sequence_length weights
        scores = self.position_weights[:sequence_length].unsqueeze(0)  # (B, L)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        # Calculate attention weights
        attention_weights = F.softmax(scores, dim=1)  # (1, L)
        attention_weights = attention_weights.expand(
            batch_size, sequence_length
        )  # (B, L)

        # Reshape attention scores to match the input tensor shape
        attention_weights = attention_weights.unsqueeze(-1)  # (B, L, 1)

        # Weighted sum
        pooled_output = torch.sum(input * attention_weights, dim=1)  # (B, D)

        if return_attention:
            return pooled_output, attention_weights
        return pooled_output


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
        nn.init.uniform_(self.query_vector, -0.1, 0.1)

    def forward(
        self,
        input: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """
        :param Tensor input: Input tensor of shape `(batch_size, sequence_length, feature_dimension)`
        :param Tensor attention_mask: Optional attention mask for padding tokens. (0 = pad)
        :param bool return_attention: Whether to return the attention weights.
        :return: The pooled output tensor, and optionally the attention weights.
        :rtype: Union[Tensor, tuple[Tensor, Tensor]]
        """

        # Compute attention scores, scaled by the square root of feature dimension
        scores = input @ self.query_vector / math.sqrt(self.feature_dimension)  # (B, L)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        # Calculate attention weights
        attention_weights = F.softmax(scores, dim=1)  # (B, L)

        # Reshape attention weights to match the input tensor shape
        attention_weights = attention_weights.unsqueeze(-1)  # (B, L, 1)

        # Weighted sum
        pooled_output = torch.sum(input * attention_weights, dim=1)  # (B, D)

        if return_attention:
            return pooled_output, attention_weights
        return pooled_output


class MultiHeadGatedSelfAttentionPooling(nn.Module):
    """
    Multi-head self-attention pooling with learned queries and token-level gating.

    Extends SelfAttentionPooling by using multiple heads and a gating mechanism
    applied to each token before pooling.

    Inspired from LSTM gating mechanisms and multi-head attention in Transformers (with learned static queries).

    Sources:
    "Structured Self-attentive Sentence Embedding" [(Lin et al., 2017)](https://arxiv.org/abs/1703.03130)
    "Language Modeling with Gated Convolutional Networks" [(Dauphin et al., 2017)](https://arxiv.org/abs/1612.08083)

    Benefits include:
    - Multi-head allows diverse pooling criteria (different semantic focuses) across heads.
    - Gating suppresses noisy/unimportant token features before pooling.
    - Learnable queries adaptively focus on relevant tokens based on content.
    """

    def __init__(self, feature_dimension: int, number_of_heads: int):
        super().__init__()

        assert feature_dimension % number_of_heads == 0, (
            f"({feature_dimension}) must be divisible by number_of_heads ({number_of_heads})."
        )

        self.feature_dimension = feature_dimension
        self.number_of_heads = number_of_heads
        self.head_dimension = feature_dimension // number_of_heads

        # Learnable query vectors for each head
        self.queries = nn.Parameter(
            torch.empty(number_of_heads, self.head_dimension)
        )  # (N, D_h)

        # Gating network, per-token gates
        self.gate = nn.Linear(feature_dimension, feature_dimension)

        # Projection to feature dimension after pooling
        self.output_projection = nn.Linear(feature_dimension, feature_dimension)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.queries, -0.1, 0.1)
        self.gate.reset_parameters()
        self.output_projection.reset_parameters()

    def forward(
        self,
        input: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """
        :param Tensor input: Input tensor of shape `(batch_size, sequence_length, feature_dimension)`
        :param Tensor attention_mask: Optional attention mask for padding tokens. (0 = pad)
        :param bool return_attention: Whether to return the attention weights.
        :return: The pooled output tensor, and optionally the attention weights.
        :rtype: Union[Tensor, tuple[Tensor, Tensor]]
        """

        batch_size, sequence_length, feature_dimension = input.shape

        # Reshape input for multi-head attention
        input_reshaped = input.reshape(
            batch_size, sequence_length, self.number_of_heads, self.head_dimension
        )  # (B, L, N, D_h)

        # Move heads forward for easy computation, swap L and N
        input_reshaped = input_reshaped.transpose(1, 2)  # (B, N, L, D_h)

        # Queries need to be broadcast-able to (B, N, L, D_h), so we add batch dimension and sequence length
        queries = self.queries.unsqueeze(0).unsqueeze(2)  # (1, N, 1, D_h)

        # Compute attention scores, scaled by the square root of head size
        scores = (input_reshaped * queries).sum(-1) / math.sqrt(
            self.head_dimension
        )  # (B, N, L)

        # Apply attention mask if provided
        if attention_mask is not None:
            # We need to expand attention_mask to match scores shape
            # attention_mask: (B, L) -> (B, 1, L)
            expanded_mask = attention_mask.unsqueeze(1)  # (B, 1, L)
            scores = scores.masked_fill(expanded_mask == 0, float("-inf"))

        # Calculate attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (B, N, L)

        # Gating mechanism, token-level gates on original input
        gates = torch.sigmoid(self.gate(input))  # (B, L, D)

        # Reshape gates for multi-head
        gates_reshaped = gates.reshape(
            batch_size, sequence_length, self.number_of_heads, self.head_dimension
        ).transpose(1, 2)  # (B, N, L, D_h)

        # Weighted sum with attention weights and gates
        # We need to expand attention_weights to match input_reshaped shape
        # attention_weights: (B, N, L) -> (B, N, L, 1)
        attention_weights_expanded = attention_weights.unsqueeze(-1)  # (B, N, L, 1)

        # Element-wise multiplication and sum over sequence length
        gated_input = input_reshaped * gates_reshaped  # (B, N, L, D_h)

        # Pooling over sequence length
        pooled = (gated_input * attention_weights_expanded).sum(dim=2)  # (B, N, D_h)

        # Concatenate heads by reshaping back to (B, D)
        pooled = pooled.reshape(batch_size, feature_dimension)  # (B, D)

        # Final projection
        output = self.output_projection(pooled)  # (B, D)

        if return_attention:
            return output, attention_weights
        return output
