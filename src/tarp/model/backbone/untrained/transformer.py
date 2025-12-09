# Transformer model for sequence classification
from typing import Optional

from torch import Tensor, nn

from tarp.model.backbone import Encoder
from tarp.model.layers.attention.multihead.rotational import (
    MultiHeadSelfAttentionWithRotaryPositionalEmbeddings,
)
from tarp.model.layers.pooling.learned import SelfAttentionPooling


class TransformerEncoderLayerWithRotaryPositionalEmbeddings(nn.Module):
    def __init__(
        self,
        number_of_heads: int,
        model_dimension: int,
        feedforward_dimension: int,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU(),
        epsilon: float = 1e-5,
        normalize_first: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadSelfAttentionWithRotaryPositionalEmbeddings(
            model_dimension=model_dimension,
            number_of_heads=number_of_heads,
            dropout=dropout,
        )

        self.feedforward_normalization = nn.LayerNorm(model_dimension, eps=epsilon)

        self.feedforward = nn.Sequential(
            nn.Linear(model_dimension, feedforward_dimension, bias=bias),
            nn.Dropout(dropout),
            activation,
            nn.Linear(feedforward_dimension, model_dimension, bias=bias),
        )

        self.attention_dropout = nn.Dropout(dropout)
        self.attention_normalization = nn.LayerNorm(model_dimension, eps=epsilon)

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
            feedforward_output = self.feedforward(normalized_states)
            hidden_states = hidden_states + feedforward_output

        # Post-norm architecture
        else:
            # Attention block with residual
            attention_output = self._self_attention_block(
                hidden_states, attention_mask, is_causal
            )
            hidden_states = hidden_states + attention_output
            hidden_states = self.attention_normalization(hidden_states)

            # Feedforward block with residual
            feedforward_output = self.feedforward(hidden_states)
            hidden_states = self.feedforward_normalization(
                hidden_states + feedforward_output
            )

        return hidden_states


class TransformerEncoder(Encoder):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int,
        feedforward_dimension: int,
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
                TransformerEncoderLayerWithRotaryPositionalEmbeddings(
                    model_dimension=embedding_dimension,
                    number_of_heads=number_of_heads,
                    feedforward_dimension=feedforward_dimension,
                    dropout=dropout,
                )
                for _ in range(number_of_layers)
            ]
        )

        self.normalization = nn.LayerNorm(embedding_dimension)
        self.dropout = nn.Dropout(dropout)
        self.embedding_dimension = embedding_dimension
        self.pooling = SelfAttentionPooling(embedding_dimension)
        self.output_dimension = embedding_dimension

    def encode(
        self,
        sequence: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_sequence: bool = False,
    ) -> Tensor:
        embeddings = self.embedding(sequence)  # (B, L, D)

        mask = None
        if attention_mask is not None:
            # attention_mask: (B, L) -> (B, 1, 1, L)
            mask = attention_mask[:, None, None, :].bool()

        encoded = embeddings
        for layer in self.transformer_encoder:
            encoded = layer(
                hidden_states=encoded,
                attention_mask=mask,
                is_causal=False,
            )

        if return_sequence:
            return self.dropout(self.normalization(encoded))
        else:
            return self.pooling(self.dropout(self.normalization(encoded)))

    @property
    def encoding_size(self) -> int:
        return self.output_dimension
