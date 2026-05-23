from collections.abc import Sequence
from typing import override

import torch
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
        maximum_span_length: int = 10,
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
        self.maximum_span_length = maximum_span_length

    @override
    def transform(self, index: int, row: dict[str, str]) -> LanguageBatch:
        raw_sequence = row[self.sequence_column]
        sequence, attention_mask = self.preprocessing(raw_sequence)
        truth = sequence.clone()

        specials_tokens = torch.tensor(
            list(self.tokenizer.special_tokens_and_ids.values()), device=sequence.device
        ).to(sequence.device, non_blocking=True)
        is_special = torch.isin(sequence, specials_tokens)
        valid_indices = torch.nonzero(~is_special).squeeze(1)

        if valid_indices.numel() == 0:
            return {
                "sequence": sequence,
                "attention_mask": attention_mask,
                "truth": truth,
            }

        adjusted_masking_probability = self.masking_probability / (
            self.maximum_span_length / 2
        )

        start_candidates = (
            torch.rand(valid_indices.size(0), device=sequence.device)
            < adjusted_masking_probability
        )
        start_indices = valid_indices[start_candidates]

        span_mask = torch.zeros_like(sequence, dtype=torch.bool)

        if start_indices.numel() > 0:
            distribution = torch.distributions.Geometric(0.2)
            span_lengths = (
                distribution.sample((start_indices.shape)).to(
                    dtype=torch.long, device=sequence.device
                )
                + 1
            )
            span_lengths = span_lengths.clamp(max=self.maximum_span_length)

            maximum_span_length = int(span_lengths.max().item())

            offsets = torch.arange(
                maximum_span_length, device=sequence.device
            ).unsqueeze(0)

            span_matrix = start_indices.unsqueeze(1) + offsets
            valid_cells = (offsets < span_lengths.unsqueeze(1)) & (
                span_matrix < sequence.size(0)
            )
            span_mask[span_matrix[valid_cells]] = True
            span_mask[is_special] = False

        masked_positions = torch.nonzero(span_mask).squeeze(1)

        if masked_positions.numel() > 0:
            random = torch.rand(masked_positions.numel(), device=sequence.device)

            is_mask_token = random < 0.8
            is_random_token = (random >= 0.8) & (random < 0.9)

            mask_token_id = self.tokenizer.mask_token_id
            sequence[masked_positions[is_mask_token]] = mask_token_id

            random_indices = masked_positions[is_random_token]
            random_tokens = torch.randint(
                low=0,
                high=self.tokenizer.vocabulary_size,
                size=(random_indices.numel(),),
                device=sequence.device,
            )
            sequence[random_indices] = random_tokens

        truth[~span_mask] = -100

        return {
            "sequence": sequence,
            "attention_mask": attention_mask,
            "truth": truth,
        }

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
