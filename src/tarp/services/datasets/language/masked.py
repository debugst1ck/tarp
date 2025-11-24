import torch
from torch import Tensor

from tarp.services.datasets import SequenceDataset
from tarp.services.datasource.sequence import SequenceDataSource
from tarp.services.preprocessing.augmentation import Augmentation, NoAugmentation
from tarp.services.tokenizers import Tokenizer


class MaskedLanguageModelDataset(SequenceDataset):
    def __init__(
        self,
        data_source: SequenceDataSource,
        tokenizer: Tokenizer,
        sequence_column: str,
        maximum_sequence_length: int,
        augmentation: Augmentation = NoAugmentation(),
        masking_probability: float = 0.15,
    ):
        super().__init__(
            data_source,
            tokenizer,
            sequence_column,
            maximum_sequence_length,
            augmentation,
        )
        self.mask_token_id = tokenizer.mask_token_id
        self.masking_probability = masking_probability

    def process_row(self, index: int, row: dict) -> dict[str, Tensor]:
        item = super().process_row(index, row)

        sequence = item["sequence"]
        attention_mask = item["attention_mask"]
        truth = sequence.clone()

        # Do not mask PAD tokens, attention mask will handle them
        probability_matrix = torch.full(
            sequence.shape, self.masking_probability, device=sequence.device
        )
        probability_matrix = probability_matrix * attention_mask

        # Get masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set unmasked positions in truth to -100 so they are ignored in loss computation
        truth[~masked_indices] = -100

        # Use BERT-style masking
        # 80% MASK, 10% random token, 10% original token

        # 80% MASK
        indices_replaced = masked_indices & (
            torch.bernoulli(torch.full(sequence.shape, 0.8)).bool()
        )

        sequence[indices_replaced] = self.mask_token_id

        # 10% get replaced with random tokens
        indices_random = (
            masked_indices
            & ~indices_replaced
            & (torch.bernoulli(torch.full(sequence.shape, 0.5)).bool())
        )
        random_words = torch.randint(
            self.tokenizer.vocab_size, sequence.shape, dtype=torch.long
        )
        sequence[indices_random] = random_words[indices_random]

        # The rest 10% are left unchanged

        return {
            "sequence": sequence,
            "attention_mask": attention_mask,
            "truth": truth,
        }
