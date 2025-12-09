from typing import Optional

from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tarp.model.backbone import Encoder
from tarp.model.layers.pooling.learned import SelfAttentionPooling


class LstmEncoder(Encoder):
    """
    An LSTM-based encoder that supports packed sequences, bi-directionality,
    and configurable layer depth.

    Produces either:
    - The per-token sequence outputs  (B, L, H)
    - A pooled final hidden state     (B, H_out)

    where H_out = hidden_dimension * (2 if bidirectional else 1)
    """

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int,
        hidden_dimension: int,
        number_of_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
        padding_id: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dimension = hidden_dimension
        self.embedding_dimension = embedding_dimension
        self.number_of_layers = number_of_layers
        self.bidirectional = bidirectional
        self.padding_id = padding_id

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dimension,
            padding_idx=padding_id,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dimension,
            hidden_size=hidden_dimension,
            num_layers=number_of_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if number_of_layers > 1 else 0.0,
        )
        self.output_dimension = (
            hidden_dimension * 2 if bidirectional else hidden_dimension
        )
        self.pooling = SelfAttentionPooling(
            self.output_dimension,
        )
        self.dropout = nn.Dropout(dropout)
        self.normalization = nn.LayerNorm(self.output_dimension)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()

        # LSTM: PyTorch-style Xavier/uniform initialization
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # LSTM forget-gate bias trick
                param.data[self.hidden_dimension : 2 * self.hidden_dimension] = 1.0

    def encode(
        self,
        sequence: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_sequence: bool = False,
    ) -> Tensor:
        embedded = self.embedding(sequence)  # (B, L, E)

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed)
            output, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=sequence.size(1),
                padding_value=self.padding_id,
            )
        else:
            output, (hidden, cell) = self.lstm(embedded)

        # Hidden : (num_layers * num_directions, B, H)
        if return_sequence:
            return output  # (B, L, H_out)
        else:
            pooled = self.pooling(output, attention_mask)  # (B, H_out)
            pooled = self.normalization(self.dropout(pooled))
            return pooled  # (B, H_out)

    @property
    def encoding_size(self) -> int:
        return self.output_dimension
