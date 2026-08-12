from collections.abc import Sequence
from typing import cast, override

import torch

from tarp.data.datasets.core import SequenceDataset
from tarp.data.sources.sequence import SequenceDataSource
from tarp.preprocessing.augmentation.core import Augmentation
from tarp.preprocessing.tokenizers.core import Tokenizer
from tarp.typed.batch import ClassificationBatch


class ClassificationDataset(
    SequenceDataset[dict[str, str | float], ClassificationBatch]
):
    def __init__(
        self,
        source: SequenceDataSource[dict[str, str | float]],
        tokenizer: Tokenizer,
        sequence_column: str,
        label_columns: Sequence[str],
        augmentation: Augmentation | None = None,
        maximum_sequence_length: int | None = 2048,
        static_sequence_length: bool = True,
    ):
        super().__init__(
            source=source,
            tokenizer=tokenizer,
            sequence_column=sequence_column,
            augmentation=augmentation,
            maximum_sequence_length=maximum_sequence_length,
            static_sequence_length=static_sequence_length,
        )
        self.label_columns: Sequence[str] = label_columns

    @override
    def transform(self, index: int, row: dict[str, str | float]) -> ClassificationBatch:
        sequence, attention_mask = self.preprocessing(
            cast(str, row[self.sequence_column])
        )
        labels = torch.as_tensor(
            [row.get(col, 0) for col in self.label_columns], dtype=torch.float
        )
        return {
            "sequence": sequence,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    @override
    def collate(self, batch: Sequence[ClassificationBatch]) -> ClassificationBatch:
        padded_seqs, padded_masks = self.pad_sequence_and_mask(
            [item["sequence"] for item in batch],
            [item["attention_mask"] for item in batch],
        )
        labels = torch.stack([item["labels"] for item in batch], dim=0)
        return ClassificationBatch(
            sequence=padded_seqs, attention_mask=padded_masks, labels=labels
        )
