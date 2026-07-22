from collections.abc import Sequence
from typing import override

import torch
from torch import Tensor

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
        if target == 0:
            return torch.empty(0, dtype=torch.long, device=positions.device)
        selected = torch.randperm(positions.numel(), device=positions.device)[:target]
        return positions[selected]

    def mask_sequence(self, sequence: Tensor, masking_positions: Tensor) -> Tensor:
        random = torch.rand(masking_positions.shape, device=sequence.device)

        # Replace 80% of the masked positions with the mask token
        is_mask_token = random < 0.8
        sequence[masking_positions[is_mask_token]] = self.tokenizer.mask_token_id

        # Replace 10% of the masked positions with random tokens
        is_random_token = (random >= 0.8) & (random < 0.9)
        random_indices = masking_positions[is_random_token]
        random_tokens = torch.randint_like(
            random_indices,
            low=0,
            high=self.tokenizer.vocabulary_size,
        )
        sequence[random_indices] = random_tokens

        # remaining 10% stay unchanged

        return sequence

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
            truth[masked_positions] = sequence[masked_positions]
            sequence = self.mask_sequence(sequence, masked_positions)

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
        padded_truths = self.sequence_padding(
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
        """Implements Span Masking protecting contiguity without destroying spans via random trim."""
        n = positions.numel()

        target_masks = int(round(self.masking_probability * n))
        if target_masks == 0:
            return torch.empty(0, dtype=torch.long, device=positions.device)

        mask = torch.zeros(n, dtype=torch.bool, device=positions.device)
        current_masks = 0

        # Sample spans sequentially until budget fulfilled to maintain span contiguity
        while current_masks < target_masks:
            span_len = int(
                torch.poisson(
                    torch.tensor([self.expected_span], device=positions.device)
                ).item()
            )
            if span_len < 1:
                span_len = 1

            if current_masks + span_len > target_masks:
                span_len = target_masks - current_masks

            start_idx = int(torch.randint(0, n, (1,)).item())
            end_idx = min(start_idx + span_len, n)

            # Count how many new tokens get masked
            new_masks = (~mask[start_idx:end_idx]).sum().item()
            mask[start_idx:end_idx] = True
            current_masks += int(new_masks)

        return positions[mask]

    @override
    def mask_sequence(self, sequence: Tensor, masking_positions: Tensor) -> Tensor:
        # In dLLM models, we typically replace all masked positions with [MASK] token
        sequence[masking_positions] = self.tokenizer.mask_token_id
        return sequence
