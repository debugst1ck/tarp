from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, Optional

from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from tarp.services.datasources.sequence import SequenceDataSource
from tarp.services.preprocessing.augmentation import Augmentation, NoAugmentation
from tarp.services.tokenizers import Tokenizer
from tarp.typing.dataset import T


class SequenceDataset(ABC, Dataset, Generic[T]):
    """
    An abstract base class for sequence datasets.
    This class provides common functionality for handling sequence data, including tokenization, attention mask creation, and collation.
    Subclasses should implement the `process_row` method to define how individual rows are processed and the `collate_function` method to define how batches are collated.
    Optionally, `__getitems__` can be overridden for optimized batch retrieval if the default implementation is not efficient enough.
    """

    def __init__(
        self,
        data_source: SequenceDataSource,
        tokenizer: Tokenizer,
        sequence_column: str,
        augmentation: Augmentation = NoAugmentation(),
        maximum_sequence_length: Optional[int] = 2048,
    ):
        self.data_source = data_source
        self.tokenizer = tokenizer
        self.sequence_column = sequence_column
        self.padding_value = tokenizer.pad_token_id
        self.augmentation = augmentation
        self.maximum_sequence_length = maximum_sequence_length

    def __len__(self):
        return self.data_source.height

    def __getitem__(self, index: int) -> T:
        """
        Retrieve a single item by its index.

        :param int index: Index of the item to retrieve.
        :return T: Processed item.
        """
        row = self.data_source.retrieve(index)
        return self.process_row(index, row)

    def __getitems__(
        self, indices: Sequence[int], rows: Optional[Sequence[dict]] = None
    ) -> Sequence[T]:
        """
        Retrieve multiple items by their indices.

        :param Sequence[int] indices: List of indices to retrieve.
        :param Optional[Sequence[dict]] rows: Optional pre-fetched rows corresponding to the indices. If provided, this will be used instead of fetching from the data source.
        :return Sequence[T]: List of processed items.
        """
        if rows is None:
            rows = self.data_source.batch(indices)
        return [self.process_row(index, row) for index, row in zip(indices, rows)]

    def preprocessing(self, row: dict) -> dict[str, Tensor]:
        """
        A common preprocessing step for sequence datasets. Can be overridden by subclasses.

        :param dict row: The data row to preprocess.
        :return dict[str, Tensor]: A dictionary containing the preprocessed sequence and attention mask.
        """
        sequence = row[self.sequence_column]
        sequence = self.augmentation.apply(sequence)
        tokenized = self.tokenizer.tokenize(sequence)

        # Attention mask
        attention_mask = tokenized != self.padding_value

        # Truncate if necessary
        if self.maximum_sequence_length is not None:
            tokenized = tokenized[: self.maximum_sequence_length]
            attention_mask = attention_mask[: self.maximum_sequence_length]

        return {"sequence": tokenized, "attention_mask": attention_mask}

    def pad_sequences_and_masks(
        self, batch: Sequence[dict[str, Tensor]]
    ) -> dict[str, Tensor]:
        """
        Pad sequences and attention masks in the batch to the same length.

        :param Sequence[dict[str, Tensor]] batch: A batch of items, each containing a 'sequence' and 'attention_mask'.
        :return dict[str, Tensor]: A dictionary with padded 'sequence' and 'attention_mask'.
        """
        sequences = [item["sequence"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]

        padded_sequences = pad_sequence(
            sequences, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        padded_attention_masks = pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )

        return {
            "sequence": padded_sequences,
            "attention_mask": padded_attention_masks,
        }

    @abstractmethod
    def process_row(self, index: int, row: dict) -> T:
        raise NotImplementedError

    @abstractmethod
    def collate_function(self, batch: Sequence[T]) -> T:
        raise NotImplementedError
