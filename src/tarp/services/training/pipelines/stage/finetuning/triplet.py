from pathlib import Path

import polars as pl
import torch
import torch.multiprocessing as mp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from tarp.cli.logging import Console
from tarp.model.backbone import Encoder
from tarp.model.finetuning.metric.triplet import TripletMetricModel
from tarp.services.datasets.classification.multilabel import (
    MultiLabelClassificationDataset,
)
from tarp.services.datasets.metric.triplet import MultiLabelOfflineTripletDataset
from tarp.services.preprocessing.augmentation import CompositeAugmentation
from tarp.services.preprocessing.augmentation.nucleotide import (
    InsertionDeletion,
    RandomMutation,
    ReverseComplement,
)
from tarp.services.tokenizers import Tokenizer
from tarp.services.training.pipelines.stage import Stage
from tarp.services.training.trainer.metric.triplet import OfflineTripletMetricTrainer


class TripletMetricFinetuningStage(Stage):
    def __init__(self, run_id: str, device: torch.device) -> None:
        super().__init__("Triplet Metric Finetuning", run_id, device)

    def run(self, encoder: Encoder, tokenizer: Tokenizer) -> Encoder:
        # Check that sources are set
        if self.train_source is None or self.valid_source is None:
            Console.error("Train and valid sources must be set before running stage")
            return encoder

        label_columns = (
            pl.read_csv(Path("temp/data/cache/labels.csv")).to_series().to_list()
        )

        triplet_dataset_train = MultiLabelOfflineTripletDataset(
            base_dataset=MultiLabelClassificationDataset(
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
            ),
            label_cache=Path("temp/data/cache/labels_cache_train.parquet"),
        )

        triplet_dataset_valid = MultiLabelOfflineTripletDataset(
            base_dataset=MultiLabelClassificationDataset(
                self.valid_source,
                tokenizer,
                sequence_column="dna_sequence",
                label_columns=label_columns,
                maximum_sequence_length=512,
            ),
            label_cache=Path("temp/data/cache/labels_cache_valid.parquet"),
        )

        triplet_model = TripletMetricModel(
            encoder=encoder,
        )
        self._model = triplet_model

        torch.compile(triplet_model, mode="max-autotune")

        multi_triplet_optimizer = AdamW(
            triplet_model.encoder.optimizer_groups(base_learning_rate=2e-5)
        )
        OfflineTripletMetricTrainer(
            model=triplet_model,
            train_dataset=triplet_dataset_train,
            valid_dataset=triplet_dataset_valid,
            optimizer=multi_triplet_optimizer,
            scheduler=CosineAnnealingWarmRestarts(
                multi_triplet_optimizer, T_0=3, T_mult=2
            ),
            device=self.device,
            epochs=6,
            num_workers=4,
            batch_size=64,
            accumulation_steps=4,
            persistent_workers=(
                True if mp.get_start_method(allow_none=True) == "spawn" else False
            ),
        ).fit()

        return triplet_model.encoder
