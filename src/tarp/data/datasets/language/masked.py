from collections.abc import Sequence
from typing import override

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from tarp.data.datasets.core import SequenceDataset
from tarp.data.sources.sequence import SequenceDataSource
from tarp.preprocessing.augmentation.core import Augmentation
from tarp.preprocessing.tokenizers.core import Tokenizer
from tarp.typed.batch import LanguageBatch


class MaskedLanguageDataset(SequenceDataset[dict[str, str], LanguageBatch]):
    def __init__(
        self,
        source: SequenceDataSource[dict[str, str]],
        tokenizer: Tokenizer,
        sequence_column: str,
        augmentation: Augmentation | None = None,
        masking_probability: float = 0.15,
        maximum_sequence_length: int | None = 2048,
    ):
        super().__init__(
            source=source,
            tokenizer=tokenizer,
            sequence_column=sequence_column,
            augmentation=augmentation,
            maximum_sequence_length=maximum_sequence_length,
        )
        self.masking_probability = masking_probability

    def masking_positions(self, positions: Tensor) -> Tensor:
        target = int(round(self.masking_probability * positions.numel()))
        selected = torch.randperm(positions.numel(), device=positions.device)[:target]
        return positions[selected]

    @override
    def transform(self, index: int, row: dict[str, str]) -> LanguageBatch:
        raw_sequence = row[self.sequence_column]
        sequence, attention_mask = self.preprocessing(raw_sequence)
        truth = torch.full_like(sequence, -100)

        special_tokens = torch.tensor(
            list(self.tokenizer.special_tokens_and_ids.values()),
            device=sequence.device,
        )
        is_special = torch.isin(sequence, special_tokens)
        valid_indices = torch.nonzero(~is_special).squeeze(1)

        if valid_indices.numel() == 0:
            return LanguageBatch(
                sequence=sequence,
                attention_mask=attention_mask,
                truth=truth,
            )

        masked_positions = self.masking_positions(valid_indices)

        if masked_positions.numel() > 0:
            random = torch.rand(masked_positions.shape, device=sequence.device)

            truth[masked_positions] = sequence[masked_positions]

            # Replace 80% of the masked positions with the mask token
            is_mask_token = random < 0.8
            sequence[masked_positions[is_mask_token]] = self.tokenizer.mask_token_id

            # Replace 10% of the masked positions with random tokens
            is_random_token = (random >= 0.8) & (random < 0.9)
            random_indices = masked_positions[is_random_token]
            random_tokens = torch.randint_like(
                random_indices,
                low=0,
                high=self.tokenizer.vocabulary_size,
            )
            sequence[random_indices] = random_tokens

            # remaining 10% stay unchanged

            # Set the truth values for the masked positions

        return LanguageBatch(
            sequence=sequence,
            attention_mask=attention_mask,
            truth=truth,
        )

    @override
    def collate(self, batch: Sequence[LanguageBatch]) -> LanguageBatch:
        sequences, masks = self.pad_sequence_and_mask(
            [item["sequence"] for item in batch],
            [item["attention_mask"] for item in batch],
        )
        padded_truths = pad_sequence(
            [item["truth"] for item in batch], batch_first=True, padding_value=-100
        )
        return LanguageBatch(
            sequence=sequences, attention_mask=masks, truth=padded_truths
        )


class PoissonSpanMaskingDataset(MaskedLanguageDataset):
    def __init__(
        self,
        source: SequenceDataSource[dict[str, str]],
        tokenizer: Tokenizer,
        sequence_column: str,
        augmentation: Augmentation | None = None,
        masking_probability: float = 0.15,
        expected_span: float = 18.0,
        maximum_sequence_length: int | None = 2048,
    ):
        super().__init__(
            source=source,
            tokenizer=tokenizer,
            sequence_column=sequence_column,
            augmentation=augmentation,
            masking_probability=masking_probability,
            maximum_sequence_length=maximum_sequence_length,
        )
        self.expected_span = expected_span

    @override
    def masking_positions(self, positions: Tensor) -> Tensor:
        n = positions.numel()
        if n == 0:
            return positions

        # Sample span lengths from a Poisson distribution for each position
        # We'll sample more spans than we need and trim later
        expected_spans = max(1, int(self.masking_probability * n / self.expected_span))

        # Randomly choose span start indices (into the positions array)
        start_indices = torch.randint(0, n, (expected_spans,), device=positions.device)

        # Sample span lengths from Poisson distribution, minimum length of 1
        span_lengths = (
            torch.poisson(
                torch.full(
                    (expected_spans,), self.expected_span, device=positions.device
                )
            )
            .long()
            .clamp(min=1)
        )

        # Build a boolean mask over positions
        mask = torch.zeros(n, dtype=torch.bool, device=positions.device)
        for start_idx, length in zip(start_indices.tolist(), span_lengths.tolist()):
            end_idx = min(start_idx + length, n)
            mask[start_idx:end_idx] = True

        target = int(self.masking_probability * n)
        masked_indices = mask.nonzero(as_tuple=True)[0]
        current = masked_indices.numel()

        if current > target:
            # Trim excess
            keep = masked_indices[
                torch.randperm(current, device=positions.device)[:target]
            ]
            mask = torch.zeros(n, dtype=torch.bool, device=positions.device)
            mask[keep] = True
        elif current < target:
            # Top up from unmasked positions
            unmasked_indices = (~mask).nonzero(as_tuple=True)[0]
            shortfall = target - current
            if unmasked_indices.numel() >= shortfall:
                extra = unmasked_indices[
                    torch.randperm(unmasked_indices.numel(), device=positions.device)[
                        :shortfall
                    ]
                ]
                mask[extra] = True

        return positions[mask]


class CosineDiffusionMaskingDataset(PoissonSpanMaskingDataset):
    def __init__(
        self,
        source: SequenceDataSource[dict[str, str]],
        tokenizer: Tokenizer,
        sequence_column: str,
        augmentation: Augmentation | None = None,
        minimum_masking=0.05,
        maximum_masking=0.95,
        maximum_sequence_length: int | None = 2048,
    ):
        super().__init__(
            source=source,
            tokenizer=tokenizer,
            sequence_column=sequence_column,
            augmentation=augmentation,
            maximum_sequence_length=maximum_sequence_length,
        )
        self.minimum_masking = minimum_masking
        self.maximum_masking = maximum_masking

    @override
    def masking_positions(self, positions: Tensor) -> Tensor:
        timestep = torch.rand(1, device=positions.device)
        schedule = torch.cos(0.5 * torch.pi * timestep)
        probability = self.minimum_masking + (
            self.maximum_masking - self.minimum_masking
        ) * (1.0 - schedule)
        original_masking_probability = self.masking_probability

        try:
            self.masking_probability = probability.item()
            return super().masking_positions(positions)
        finally:
            self.masking_probability = original_masking_probability
