from collections.abc import Sequence
from typing import final, override

from tarp.data.datasets.core import SequenceDataset
from tarp.data.datasets.language.masked import MaskedLanguageDataset
from tarp.data.sources.sequence import SequenceDataSource
from tarp.preprocessing.augmentation.core import Augmentation
from tarp.preprocessing.tokenizers.core import Tokenizer
from tarp.typed.batch import DistillationBatch, SequenceBatch


@final
class CrossDistillationDataset(SequenceDataset[dict[str, str], DistillationBatch]):
    def __init__(
        self,
        student_dataset: MaskedLanguageDataset,
        teacher_source: SequenceDataSource[dict[str, str]],
        teacher_tokenizer: Tokenizer,
        sequence_column: str,
        teacher_augmentation: Augmentation | None = None,
        maximum_sequence_length: int | None = 2048,
        static_sequence_length: bool = True,
    ) -> None:
        super().__init__(
            source=teacher_source,
            tokenizer=teacher_tokenizer,
            sequence_column=sequence_column,
            augmentation=teacher_augmentation,
            maximum_sequence_length=maximum_sequence_length,
            static_sequence_length=static_sequence_length,
        )
        self.student_dataset = student_dataset

    @override
    def transform(self, index: int, row: dict[str, str]) -> DistillationBatch:
        student_batch = self.student_dataset[index]
        sequence, attention_mask = self.preprocessing(row[self.sequence_column])
        teacher_batch = SequenceBatch(sequence=sequence, attention_mask=attention_mask)
        return DistillationBatch(student=student_batch, teacher=teacher_batch)

    @override
    def collate(self, batch: Sequence[DistillationBatch]) -> DistillationBatch:
        teacher_sequences, teacher_padded_masks = self.pad_sequence_and_mask(
            [item["teacher"]["sequence"] for item in batch],
            [item["teacher"]["attention_mask"] for item in batch],
        )

        # We use the collate function of the student dataset to collate the student batches
        student_batches = tuple(item["student"] for item in batch)
        student_batch = self.student_dataset.collate(student_batches)

        return DistillationBatch(
            student=student_batch,
            teacher=SequenceBatch(
                sequence=teacher_sequences, attention_mask=teacher_padded_masks
            ),
        )
