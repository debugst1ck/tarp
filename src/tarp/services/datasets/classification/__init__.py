from torch import Tensor
import torch

from tarp.services.preprocessing.augmentation import Augmentation, NoAugmentation
from tarp.services.datasets import SequenceDataset
from tarp.services.datasource.sequence import SequenceDataSource
from tarp.services.tokenizers import Tokenizer

from typing import Optional


class ClassificationDataset(SequenceDataset):
    def __init__(
        self,
        data_source: SequenceDataSource,
        tokenizer: Tokenizer,
        sequence_column: str,
        label_columns: list[str],
        maximum_sequence_length: int,
        augmentation: Augmentation = NoAugmentation(),
    ):
        super().__init__(
            data_source,
            tokenizer,
            sequence_column,
            maximum_sequence_length,
            augmentation,
        )
        self.label_columns = label_columns

    def process_row(self, index: int, row: dict) -> dict[str, Tensor]:
        item = super().process_row(index, row)
        # Extract labels for multi-source multi-label classification
        labels = [row.get(col, 0) for col in self.label_columns]
        item["labels"] = torch.tensor(labels, dtype=torch.float)
        return item