from collections.abc import Sequence
from typing import override

import torch

from tarp.data.datasets.core import SequenceDataset
from tarp.data.datasets.language.masked import PoissonSpanMaskingDataset
from tarp.data.sources.sequence import SequenceDataSource
from tarp.preprocessing.augmentation.core import Augmentation
from tarp.preprocessing.tokenizers.core import Tokenizer
from tarp.typed.batch import DiffusionBatch, LanguageBatch


class CosineDiffusionMaskingDataset(SequenceDataset[dict[str, str], DiffusionBatch]):
    def __init__(
        self,
        source: SequenceDataSource[dict[str, str]],
        tokenizer: Tokenizer,
        sequence_column: str,
        masking_probability_minimum: float = 0.05,
        masking_probability_maximum: float = 1.0,
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

        self.language_dataset: PoissonSpanMaskingDataset = PoissonSpanMaskingDataset(
            source=source,
            tokenizer=tokenizer,
            sequence_column=sequence_column,
            augmentation=augmentation,
            maximum_sequence_length=maximum_sequence_length,
            static_sequence_length=static_sequence_length,
        )
        self.masking_probability_minimum: float = masking_probability_minimum
        self.masking_probability_maximum: float = masking_probability_maximum

    @override
    def transform(self, index: int, row: dict[str, str]) -> DiffusionBatch:
        timestep = torch.rand(())

        schedule = 1.0 - torch.cos(0.5 * torch.pi * timestep)

        span = self.masking_probability_maximum - self.masking_probability_minimum
        probability = self.masking_probability_minimum + (schedule * span)

        self.language_dataset.masking_probability = probability.item()

        language_batch = self.language_dataset.transform(index, row)

        return DiffusionBatch(
            sequence=language_batch["sequence"],
            attention_mask=language_batch["attention_mask"],
            truth=language_batch["truth"],
            timestep=timestep,
        )

    @override
    def collate(self, batch: Sequence[DiffusionBatch]) -> DiffusionBatch:
        # We can kind of cheat here and use the collate function from the language dataset
        language_batches = [
            LanguageBatch(
                sequence=batch_item["sequence"],
                attention_mask=batch_item["attention_mask"],
                truth=batch_item["truth"],
            )
            for batch_item in batch
        ]
        language_batch = self.language_dataset.collate(language_batches)
        timesteps = torch.stack([batch_item["timestep"] for batch_item in batch], dim=0)

        return DiffusionBatch(
            sequence=language_batch["sequence"],
            attention_mask=language_batch["attention_mask"],
            truth=language_batch["truth"],
            timestep=timesteps,
        )
