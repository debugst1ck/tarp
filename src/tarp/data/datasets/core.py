from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from functools import partial
from typing import override

from torch import Tensor
from torch.utils.data import Dataset

from tarp.data.sources.sequence import SequenceDataSource
from tarp.functional.padding.sequence import blocked_pad_sequence, pad_to_length
from tarp.preprocessing.augmentation.core import Augmentation, NoAugmentation
from tarp.preprocessing.tokenizers.core import Tokenizer
from tarp.typed.core import KnownT


class SequenceDataset[RowT: Mapping[str, KnownT], BatchT](ABC, Dataset[BatchT]):
    def __init__(
        self,
        source: SequenceDataSource[RowT],
        tokenizer: Tokenizer,
        sequence_column: str,
        augmentation: Augmentation | None = None,
        maximum_sequence_length: int | None = 2048,
        static_sequence_length: bool = True,
    ):
        self.source: SequenceDataSource[RowT] = source
        self.tokenizer: Tokenizer = tokenizer
        self.augmentation: Augmentation = augmentation or NoAugmentation()
        self.sequence_column: str = sequence_column
        self.maximum_sequence_length: int | None = maximum_sequence_length
        self.padding_value: int = tokenizer.pad_token_id

        if static_sequence_length and maximum_sequence_length is not None:
            self.sequence_padding = partial(
                pad_to_length,
                length=maximum_sequence_length,
            )
        else:
            self.sequence_padding = blocked_pad_sequence

    def __len__(self) -> int:
        return self.source.height

    @override
    def __getitem__(self, index: int) -> BatchT:
        row = self.source.retrieve(index)
        return self.transform(index, row)

    def __getitems__(
        self,
        indices: Sequence[int],
        prefetched_rows: Sequence[RowT] | None = None,
    ) -> Sequence[BatchT]:
        rows = (
            self.source.batch(indices) if prefetched_rows is None else prefetched_rows
        )
        return [self.transform(index, row) for index, row in zip(indices, rows)]

    def preprocessing(self, sequence: str) -> tuple[Tensor, Tensor]:
        sequence = self.augmentation.apply(sequence)
        tokenized = self.tokenizer.encode(sequence)
        if self.maximum_sequence_length is not None:
            tokenized = tokenized[: self.maximum_sequence_length]
        attention_mask = tokenized != self.padding_value
        return tokenized, attention_mask

    def pad_sequence_and_mask(
        self, sequences: Sequence[Tensor], attention_masks: Sequence[Tensor]
    ) -> tuple[Tensor, Tensor]:
        padded_sequences = self.sequence_padding(
            list(sequences),
            padding_value=self.padding_value,
        )
        padded_attention_masks = self.sequence_padding(
            list(attention_masks), padding_value=0
        )
        return padded_sequences, padded_attention_masks

    @abstractmethod
    def transform(self, index: int, row: RowT) -> BatchT:
        raise NotImplementedError

    @abstractmethod
    def collate(self, batch: Sequence[BatchT]) -> BatchT:
        raise NotImplementedError
