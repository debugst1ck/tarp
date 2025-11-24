from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset

from tarp.services.datasource.sequence import SequenceDataSource
from tarp.services.preprocessing.augmentation import Augmentation, NoAugmentation
from tarp.services.tokenizers import Tokenizer


class SequenceDataset(Dataset):
    def __init__(
        self,
        data_source: SequenceDataSource,
        tokenizer: Tokenizer,
        sequence_column: str,
        maximum_sequence_length: int,
        augmentation: Augmentation = NoAugmentation(),
    ):
        self.data_source = data_source
        self.tokenizer = tokenizer
        self.sequence_column = sequence_column
        self.maximum_sequence_length = maximum_sequence_length
        self.padding_value = tokenizer.pad_token_id
        self.augmentation = augmentation

    def __len__(self):
        return self.data_source.height

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """
        Retrieve a single item by its index.

        :param index: Index of the item to retrieve.
        :param row: Optional pre-fetched row corresponding to the index. If provided, this will be used instead of fetching from the data source.
        :return: Processed item.
        """
        row = self.data_source.retrieve(index)
        return self.process_row(index, row)

    def __getitems__(
        self, indices: list[int], rows: Optional[list[dict]] = None
    ) -> list[dict[str, Tensor]]:
        """
        Retrieve multiple items by their indices.

        :param indices: List of indices to retrieve.
        :param rows: Optional pre-fetched rows corresponding to the indices. If provided, this will be used instead of fetching from the data source.
        :return: List of processed items.
        """
        if rows is None:
            rows = self.data_source.batch(indices)
        return [self.process_row(index, row) for index, row in zip(indices, rows)]

    def process_row(self, index: int, row: dict) -> dict[str, Tensor]:
        sequence = row[self.sequence_column]
        sequence = self.augmentation.apply(sequence)
        tokenized = self.tokenizer.tokenize(sequence)

        # Get sequence length from tensor
        sequence_length = tokenized.size(0)

        # Pad tokenized to maximum sequence length
        padded = torch.full(
            (self.maximum_sequence_length,), self.padding_value, dtype=torch.long
        )
        length = min(sequence_length, self.maximum_sequence_length)
        padded[:length] = tokenized[:length]

        # Attention mask
        attention_mask = padded != self.padding_value

        return {"sequence": padded, "attention_mask": attention_mask}
