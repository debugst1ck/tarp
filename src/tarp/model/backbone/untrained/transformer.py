# Transformer model for sequence classification
from torch import nn, Tensor
import torch
import torch.nn.functional as F

from tarp.model.backbone import Encoder
from typing import Optional, Callable

from tarp.model.layers.pooling.trainable import QueryAttentionPooling

from tarp.model.layers.attention.multihead.rotational import MultiHeadSelfAttentionWithRoPE

class TransformerEncoderLayerWithRoPE(nn.Module):
    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int,
        feedforward_dimension: int,
        dropout: float = 0.1,
        activation: Callable = F.relu,
        normalization_epsilon: float = 1e-5,
        normalize_first: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.self_attention = MultiHeadSelfAttentionWithRoPE(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            dropout=dropout,
        )
        self.feedforward_1 = nn.Linear(
            embedding_dimension, feedforward_dimension, bias=bias
        )
        self.feedforward_2 = nn.Linear(
            feedforward_dimension, embedding_dimension, bias=bias
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward_dropout = nn.Dropout(dropout)
        self.attention_normalization = nn.LayerNorm(
            embedding_dimension, eps=normalization_epsilon
        )
        self.feedforward_normalization = nn.LayerNorm(
            embedding_dimension, eps=normalization_epsilon
        )

        self.activation = activation
        self.normalize_first = normalize_first

    def _self_attention_block(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        attention_output = self.self_attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attention_mask=attention_mask,
            is_causal=is_causal,
        )
        return self.attention_dropout(attention_output)

    def _feedforward_block(self, hidden_states: Tensor) -> Tensor:
        feedforward_output = self.feedforward_1(hidden_states)
        feedforward_output = self.activation(feedforward_output)
        feedforward_output = self.feedforward_dropout(feedforward_output)
        feedforward_output = self.feedforward_2(feedforward_output)
        return feedforward_output

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        if self.normalize_first:
            # Attention block with residual
            normalized_states = self.attention_normalization(hidden_states)
            attention_output = self._self_attention_block(
                normalized_states, attention_mask, is_causal
            )
            hidden_states = hidden_states + attention_output

            # Feedforward block with residual
            normalized_states = self.feedforward_normalization(hidden_states)
            feedforward_output = self._feedforward_block(normalized_states)
            hidden_states = hidden_states + feedforward_output

        # Post-norm architecture
        else:
            # Attention block with residual
            attention_output = self._self_attention_block(
                hidden_states, attention_mask, is_causal
            )
            hidden_states = self.attention_normalization(
                hidden_states + attention_output
            )

            # Feedforward block with residual
            feedforward_output = self._feedforward_block(hidden_states)
            hidden_states = self.feedforward_normalization(
                hidden_states + feedforward_output
            )

        return hidden_states


class TransformerEncoder(Encoder):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int,
        hidden_dimension: int,
        number_of_layers: int = 2,
        number_of_heads: int = 4,
        dropout: float = 0.1,
        padding_id: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dimension,
            padding_idx=padding_id,
        )
        self.transformer_encoder = nn.ModuleList(
            [
                TransformerEncoderLayerWithRoPE(
                    embedding_dimension=embedding_dimension,
                    number_of_heads=number_of_heads,
                    feedforward_dimension=hidden_dimension,
                    dropout=dropout,
                )
                for _ in range(number_of_layers)
            ]
        )

        self.normalization = nn.LayerNorm(embedding_dimension)
        self.dropout = nn.Dropout(dropout)
        self.embedding_dimension = embedding_dimension
        self.pooling = QueryAttentionPooling(embedding_dimension)
        self.output_dimension = embedding_dimension

    def encode(
        self,
        sequence: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_sequence: bool = False,
    ) -> Tensor:
        embeddings = self.embedding(sequence)  # (batch, seq_len, embedding_dimension)

        encoded = embeddings
        for layer in self.transformer_encoder:
            encoded = layer(
                hidden_states=encoded,
                attention_mask=attention_mask,
                is_causal=False,
            )

        if return_sequence:
            return self.dropout(self.normalization(encoded))
        else:
            return self.pooling(self.dropout(self.normalization(encoded)))

    @property
    def encoding_size(self) -> int:
        return self.output_dimension
