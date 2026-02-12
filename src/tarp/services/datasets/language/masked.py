from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from tarp.services.datasets import SequenceDataset
from tarp.services.datasources.sequence import SequenceDataSource
from tarp.services.preprocessing.augmentation import Augmentation, NoAugmentation
from tarp.services.tokenizers import Tokenizer


class MaskedLanguageModelDataset(SequenceDataset[dict[str, Tensor]]):
    def __init__(
        self,
        data_source: SequenceDataSource,
        tokenizer: Tokenizer,
        sequence_column: str,
        augmentation: Augmentation = NoAugmentation(),
        masking_probability: float = 0.15,
    ):
        super().__init__(
            data_source,
            tokenizer,
            sequence_column,
            augmentation,
        )
        self.mask_token_id = tokenizer.mask_token_id
        self.masking_probability = masking_probability

    def process_row(self, index: int, row: dict) -> dict[str, Tensor]:
        item = self.preprocessing(row)

        sequence = item["sequence"]
        attention_mask = item["attention_mask"]
        truth = sequence.clone()

        # Do not mask PAD tokens, attention mask will handle them
        probability_matrix = torch.full(
            sequence.shape, self.masking_probability, device=sequence.device
        )

        # Mask out all special tokens (e.g., CLS, SEP, MASK) by setting their masking probability to 0
        specials = self.tokenizer.special_tokens_and_ids

        for token_id in specials.values():
            # Remove MASK token from specials to allow it to be masked if it appears in the input
            # This is important for models like BERT where the MASK token can appear in the input and should be masked
            if token_id == self.mask_token_id:
                continue
            special_token_mask = sequence == token_id
            probability_matrix[special_token_mask] = 0.0

        # Get masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set unmasked positions in truth to -100 so they are ignored in loss computation
        truth[~masked_indices] = -100

        # Use BERT-style masking
        # 80% MASK, 10% random token, 10% original token

        # 80% MASK
        indices_replaced = masked_indices & (
            torch.bernoulli(
                torch.full(sequence.shape, 0.8, device=sequence.device)
            ).bool()
        )

        sequence[indices_replaced] = self.mask_token_id

        # 10% get replaced with random tokens
        indices_random = (
            masked_indices
            & ~indices_replaced
            & (
                torch.bernoulli(
                    torch.full(sequence.shape, 0.5, device=sequence.device)
                ).bool()
            )
        )
        random_words = torch.randint(
            self.tokenizer.vocab_size,
            sequence.shape,
            dtype=torch.long,
            device=sequence.device,
        )
        sequence[indices_random] = random_words[indices_random]

        # The rest 10% are left unchanged

        return {
            "sequence": sequence,
            "attention_mask": attention_mask,
            "truth": truth,
        }

    def collate_function(self, batch: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
        padded_data = self.pad_sequences_and_masks(batch)

        # "truth": truth,
        # Truth is also same as input sequence but with unmasked tokens set to -100 so it also needs to be padded with pad_token_id
        truths = [item["truth"] for item in batch]
        padded_truths = pad_sequence(truths, batch_first=True, padding_value=-100)
        padded_data["truth"] = padded_truths

        return padded_data
