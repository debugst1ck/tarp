from pathlib import Path

import polars as pl
import torch
import torch.multiprocessing as mp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from tarp.cli.logging import Console
from tarp.model.backbone import Encoder
from tarp.model.finetuning.classification import ClassificationModel
from tarp.services.datasets.classification.multilabel import (
    MultiLabelClassificationDataset,
)
from tarp.services.evaluation.losses.multilabel import AsymmetricFocalLoss
from tarp.services.preprocessing.augmentation import CompositeAugmentation
from tarp.services.preprocessing.augmentation.nucleotide import (
    InsertionDeletion,
    RandomMutation,
    ReverseComplement,
)
from tarp.services.tokenizers import Tokenizer
from tarp.services.training.pipelines.stage import Stage
from tarp.services.training.trainer.classification.multilabel import (
    MultiLabelClassificationTrainer,
)
from tarp.utilities.operations import rescale


class MultiLabelClassificationFinetuningStage(Stage):
    def __init__(self, run_id: str, device: torch.device) -> None:
        super().__init__("Multi-Label Classification Fine-tuning", run_id, device)

    def run(self, encoder: Encoder, tokenizer: Tokenizer) -> Encoder:
        # Check that sources are set
        if self.train_source is None or self.valid_source is None:
            Console.error("Train and valid sources must be set before running stage")
            return encoder

        label_columns = (
            pl.read_csv(Path("temp/data/cache/labels.csv")).to_series().to_list()
        )

        multilabel_classification_train = MultiLabelClassificationDataset(
            self.train_source,
            tokenizer,
            sequence_column="dna_sequence",
            label_columns=label_columns,
            maximum_sequence_length=512,
            augmentation=CompositeAugmentation(
                [
                    RandomMutation(),
                    InsertionDeletion(),
                    ReverseComplement(0.5),
                ]
            ),
        )

        multilabel_classification_valid = MultiLabelClassificationDataset(
            self.valid_source,
            tokenizer,
            sequence_column="dna_sequence",
            label_columns=label_columns,
            maximum_sequence_length=512,
        )

        classification_model = ClassificationModel(
            encoder=encoder,
            number_of_classes=len(label_columns),
        )
        self._model = classification_model

        torch.compile(classification_model, mode="max-autotune")

        optimizer_classification = AdamW(
            [
                *classification_model.encoder.optimizer_groups(base_learning_rate=1e-5),
                {
                    "params": classification_model.classification_head.parameters(),
                    "lr": 1e-3,
                    "weight_decay": 1e-2,
                },
            ]
        )

        label_cache = pl.read_parquet(
            Path("temp/data/cache/labels_cache_train.parquet")
        )

        assert set(label_columns).issubset(set(label_cache.columns)), (
            "Label columns missing in label cache"
        )

        label_tensor = torch.tensor(label_cache.select(label_columns).to_numpy())

        class_counts = label_tensor.sum(dim=0)
        total_counts = label_tensor.size(0)
        class_weights = (total_counts - class_counts) / class_counts
        class_weights = rescale(class_weights, 0.1, 5.0)

        Console.debug(
            str(
                pl.DataFrame(
                    {
                        "label": label_columns,
                        "class_weights": class_weights.tolist(),
                    }
                )
            )
        )

        trainer = MultiLabelClassificationTrainer(
            model=classification_model,
            train_dataset=multilabel_classification_train,
            valid_dataset=multilabel_classification_valid,
            optimizer=optimizer_classification,
            scheduler=CosineAnnealingWarmRestarts(
                optimizer_classification, T_0=5, T_mult=2
            ),
            criterion=AsymmetricFocalLoss(
                gamma_neg=1, gamma_pos=3, class_weights=class_weights
            ),
            device=self.device,
            epochs=15,
            num_workers=4,
            batch_size=64,
            accumulation_steps=4,
            persistent_workers=(
                True if mp.get_start_method(allow_none=True) == "spawn" else False
            ),
        )
        trainer.fit()
        return classification_model.encoder
