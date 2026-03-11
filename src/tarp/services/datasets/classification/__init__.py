from collections.abc import Sequence
from typing import Optional

import torch
from torch import Tensor

from tarp.services.datasets import SequenceDataset
from tarp.services.datasources.sequence import SequenceDataSource
from tarp.services.preprocessing.augmentation import Augmentation, NoAugmentation
from tarp.services.tokenizers import Tokenizer


class ClassificationDataset(SequenceDataset[dict[str, Tensor]]):
    def __init__(
        self,
        data_source: SequenceDataSource,
        tokenizer: Tokenizer,
        sequence_column: str,
        label_columns: Sequence[str],
        augmentation: Augmentation = NoAugmentation(),
        maximum_sequence_length: Optional[int] = 2048,
    ):
        super().__init__(
            data_source,
            tokenizer,
            sequence_column,
            augmentation,
            maximum_sequence_length,
        )
        self.label_columns = label_columns

    def process_row(self, index: int, row: dict) -> dict[str, Tensor]:
        item = self.preprocessing(row)
        # Extract labels for multi-source multi-label classification
        labels = torch.as_tensor(
            [row.get(col, 0) for col in self.label_columns], dtype=torch.float32
        )
        item["labels"] = labels
        return item

    def collate_function(self, batch: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
        # Pad sequences and attention masks
        padded_data = self.pad_sequences_and_masks(batch)

        # Stack labels
        labels = torch.stack([item["labels"] for item in batch], dim=0)
        padded_data["labels"] = labels

        return padded_data
