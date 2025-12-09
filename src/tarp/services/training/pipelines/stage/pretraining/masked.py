import torch
import torch.multiprocessing as mp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from tarp.cli.logging import Console
from tarp.model.backbone import Encoder
from tarp.model.finetuning.language import LanguageModel
from tarp.services.datasets.language.masked import MaskedLanguageModelDataset
from tarp.services.preprocessing.augmentation import CompositeAugmentation
from tarp.services.preprocessing.augmentation.nucleotide import (
    InsertionDeletion,
    RandomMutation,
    ReverseComplement,
)
from tarp.services.tokenizers import Tokenizer
from tarp.services.training.pipelines.stage import Stage
from tarp.services.training.trainer.language.masked import MaskedLanguageModelTrainer


class MaskedLanguageModelPretrainingStage(Stage):
    def __init__(self, run_id: str, device: torch.device) -> None:
        super().__init__("Masked Language Model Pretraining", run_id, device)

    def run(self, encoder: Encoder, tokenizer: Tokenizer) -> Encoder:
        # Check that sources are set
        if self.train_source is None or self.valid_source is None:
            Console.error("Train and valid sources must be set before running stage")
            return encoder

        masked_language_dataset_train = MaskedLanguageModelDataset(
            data_source=self.train_source,
            tokenizer=tokenizer,
            sequence_column="dna_sequence",
            maximum_sequence_length=512,
            masking_probability=0.15,
            augmentation=CompositeAugmentation(
                [
                    RandomMutation(),
                    InsertionDeletion(),
                    ReverseComplement(0.5),
                ]
            ),
        )

        masked_language_dataset_valid = MaskedLanguageModelDataset(
            data_source=self.valid_source,
            tokenizer=tokenizer,
            sequence_column="dna_sequence",
            maximum_sequence_length=512,
            masking_probability=0.15,
        )

        language_model = LanguageModel(
            encoder=encoder,
            vocabulary_size=masked_language_dataset_train.tokenizer.vocab_size,
        )
        self._model = language_model

        torch.compile(language_model, mode="max-autotune")

        optimizer_language = AdamW(
            [
                *language_model.encoder.optimizer_groups(base_learning_rate=1e-4),
                {
                    "params": language_model.language_head.parameters(),
                    "lr": 1e-3,
                    "weight_decay": 1e-2,
                },
            ]
        )

        MaskedLanguageModelTrainer(
            model=language_model,
            train_dataset=masked_language_dataset_train,
            valid_dataset=masked_language_dataset_valid,
            optimizer=optimizer_language,
            scheduler=CosineAnnealingWarmRestarts(optimizer_language, T_0=5, T_mult=2),
            device=self.device,
            epochs=20,
            num_workers=4,
            batch_size=64,
            accumulation_steps=4,
            persistent_workers=(
                True if mp.get_start_method(allow_none=True) == "spawn" else False
            ),
            vocabulary_size=masked_language_dataset_train.tokenizer.vocab_size,
        ).fit()

        return language_model.encoder
