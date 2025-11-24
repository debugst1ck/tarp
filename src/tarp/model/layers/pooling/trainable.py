import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LearnedPositionPooling(nn.Module):
    """
    Pools a sequence of embeddings into a single vector using learnable position-based weights.
    Each position has a learned scalar importance, independent of content.
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
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """
        :param Tensor hidden_states: Output from the transformer model.
        :param Tensor attention_mask: Optional attention mask for padding tokens. (0 = pad)
        :param bool return_attention: Whether to return the attention weights.
        :return: The pooled output tensor, and optionally the attention weights.
        :rtype: Union[Tensor, tuple[Tensor, Tensor]]
        """
        batch_size, sequence_length, _ = hidden_states.shape

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
        pooled_output = torch.sum(hidden_states * attention_weights, dim=1)  # (B, H)

        if return_attention:
            return pooled_output, attention_weights
        return pooled_output


class QueryAttentionPooling(nn.Module):
    """
    Pools a sequence of embeddings into a single vector using a learned query vector.
    Each token's importance is determined dynamically based on its content.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.query_vector = nn.Parameter(torch.empty(hidden_size))  # (H,)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.query_vector, -0.1, 0.1)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """
        :param Tensor hidden_states: Output from the transformer model.
        :param Tensor attention_mask: Optional attention mask for padding tokens. (0 = pad)
        :param bool return_attention: Whether to return the attention weights.
        :return: The pooled output tensor, and optionally the attention weights.
        :rtype: Union[Tensor, tuple[Tensor, Tensor]]
        """

        # Compute attention scores, scaled by the square root of hidden size
        scores = (
            hidden_states @ self.query_vector / math.sqrt(self.hidden_size)
        )  # (B, L)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        # Calculate attention weights
        attention_weights = F.softmax(scores, dim=1)  # (B, L)

        # Reshape attention weights to match the input tensor shape
        attention_weights = attention_weights.unsqueeze(-1)  # (B, L, 1)

        # Weighted sum
        pooled_output = torch.sum(hidden_states * attention_weights, dim=1)  # (B, H)

        if return_attention:
            return pooled_output, attention_weights
        return pooled_output
