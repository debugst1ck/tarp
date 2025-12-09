import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import polars as pl
import torch
import torch.multiprocessing as mp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from tarp.cli.logging import Console
from tarp.config import HyenaConfig, TransformerConfig
from tarp.model.backbone import Encoder
from tarp.model.backbone.untrained.hyena import HyenaEncoder
from tarp.model.backbone.untrained.transformer import TransformerEncoder
from tarp.model.finetuning.classification import ClassificationModel
from tarp.model.finetuning.language import LanguageModel
from tarp.model.finetuning.metric.triplet import TripletMetricModel
from tarp.services.datasets import SequenceDataSource
from tarp.services.datasets.classification.multilabel import (
    MultiLabelClassificationDataset,
)
from tarp.services.datasets.language.masked import MaskedLanguageModelDataset
from tarp.services.datasets.metric.triplet import MultiLabelOfflineTripletDataset
from tarp.services.datasource.sequence import FastaSliceSource, TabularSequenceSource
from tarp.services.evaluation.losses.multilabel import AsymmetricFocalLoss
from tarp.services.preprocessing.augmentation import CompositeAugmentation
from tarp.services.preprocessing.augmentation.nucleotide import (
    InsertionDeletion,
    RandomMutation,
    ReverseComplement,
)
from tarp.services.tokenizers.pretrained.dnabert import Dnabert2Tokenizer
from tarp.services.training.trainer.classification.multilabel import (
    MultiLabelClassificationTrainer,
)
from tarp.services.training.trainer.language.masked import MaskedLanguageModelTrainer
from tarp.services.training.trainer.metric.triplet import OfflineTripletMetricTrainer
from tarp.services.utilities.seed import establish_random_seed
from tarp.utilities.operations import rescale


class Stage(ABC):
    def __init__(self, name: str, run_id: str, device: torch.device) -> None:
        self.name = name
        self.run_id = run_id
        self.device = device
        self.params: dict = {}
        self.train_source: Optional[SequenceDataSource] = None
        self.valid_source: Optional[SequenceDataSource] = None
        self._model: Optional[torch.nn.Module] = None

    def with_sources(
        self, train_source: SequenceDataSource, valid_source: SequenceDataSource
    ) -> "Stage":
        self.train_source = train_source
        self.valid_source = valid_source
        return self

    def with_parameters(self, **params) -> "Stage":
        self.params = params
        return self

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            raise ValueError("Model has not been trained yet")
        return self._model

    @abstractmethod
    def run(self, encoder: Encoder) -> Encoder:
        raise NotImplementedError


class Pipeline:
    def __init__(self, encoder: Encoder, run_id: str) -> None:
        self.encoder = encoder
        self.run_id = run_id
        self.upcoming_stages: list[Stage] = []
        self.last_completed_stage: Optional[Stage] = None

    def __rshift__(self, other: Stage) -> "Pipeline":
        return self.pipe(other)

    def pipe(self, stage: Stage) -> "Pipeline":
        self.upcoming_stages.append(stage)
        return self

    def run(self) -> Encoder:
        current_encoder = self.encoder
        # Run upcoming stages in sequence and store completed stages removing them from upcoming
        # stages this way we can keep track of what has been done
        while self.upcoming_stages:
            stage = self.upcoming_stages.pop(0)
            Console.info(f"Starting stage: {stage.name}")
            current_encoder = stage.run(current_encoder)
            self.last_completed_stage = stage
            Console.info(f"Completed stage: {stage.name}")
        self.encoder = current_encoder
        return current_encoder


class MaskedLanguageModelPretrainingStage(Stage):
    def __init__(self, run_id: str, device: torch.device) -> None:
        super().__init__("Masked Language Model Pretraining", run_id, device)

    def run(self, encoder: Encoder) -> Encoder:
        # Check that sources are set
        if self.train_source is None or self.valid_source is None:
            Console.error("Train and valid sources must be set before running stage")
            return encoder

        masked_language_dataset_train = MaskedLanguageModelDataset(
            data_source=self.train_source,
            tokenizer=Dnabert2Tokenizer(),
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
            tokenizer=Dnabert2Tokenizer(),
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


class MultiLabelClassificationFinetuningStage(Stage):
    def __init__(self, run_id: str, device: torch.device) -> None:
        super().__init__("Multi-Label Classification Fine-tuning", run_id, device)

    def run(self, encoder: Encoder) -> Encoder:
        # Check that sources are set
        if self.train_source is None or self.valid_source is None:
            Console.error("Train and valid sources must be set before running stage")
            return encoder

        label_columns = (
            pl.read_csv(Path("temp/data/cache/labels.csv")).to_series().to_list()
        )

        multilabel_classification_train = MultiLabelClassificationDataset(
            self.train_source,
            Dnabert2Tokenizer(),
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
            Dnabert2Tokenizer(),
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


class TripletMetricFinetuningStage(Stage):
    def __init__(self, run_id: str, device: torch.device) -> None:
        super().__init__("Triplet Metric Finetuning", run_id, device)

    def run(self, encoder: Encoder) -> Encoder:
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
                Dnabert2Tokenizer(),
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
                Dnabert2Tokenizer(),
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


def save_model(model: torch.nn.Module, name: str, run_id: str) -> Path:
    path = Path(f"temp/checkpoints/{run_id}/{name}_{run_id}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Move model to CPU before saving to avoid CUDA initialization issues on load
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)
    return path


def load_model(
    model: torch.nn.Module, name: str, run_id: str, device: torch.device
) -> torch.nn.Module:
    path = Path(f"temp/checkpoints/{run_id}/{name}_{run_id}.pt")
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def main() -> None:
    Console.info("App started")

    try:
        mp.set_start_method(method="spawn", force=True)
        Console.info("Multiprocessing start method set to 'spawn'")
    except RuntimeError:
        Console.warning("Multiprocessing start method was already set, skipping...")
        pass

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set seed for reproducibility
    SEED = establish_random_seed(69420)  # FuNnY NuMbEr :D
    Console.info(f"Random seed set to {SEED}")

    encoder = TransformerEncoder(
        vocabulary_size=Dnabert2Tokenizer().vocab_size,
        embedding_dimension=TransformerConfig.embedding_dimension,
        feedforward_dimension=TransformerConfig.feedforward_dimension,
        padding_id=Dnabert2Tokenizer().pad_token_id,
        number_of_layers=TransformerConfig.number_of_layers,
        number_of_heads=TransformerConfig.number_of_heads,
        dropout=TransformerConfig.dropout,
    )

    # encoder = HyenaEncoder(
    #     vocabulary_size=Dnabert2Tokenizer().vocab_size,
    #     model_dimension=HyenaConfig.model_dimension,
    #     padding_id=Dnabert2Tokenizer().pad_token_id,
    #     number_of_layers=HyenaConfig.number_of_layers,
    #     number_of_heads=HyenaConfig.number_of_heads,
    #     recurrence_depth=HyenaConfig.recurrence_depth,
    #     mixing_width=HyenaConfig.mixing_width,
    #     local_context_size=HyenaConfig.local_context_size,
    #     dropout=HyenaConfig.dropout,
    # )

    # Training
    Console.debug(f"Model architecture: {encoder.__class__.__name__}")

    # Number of parameters
    Console.debug(
        f"Number of parameters: {sum(p.numel() for p in encoder.parameters() if p.requires_grad)}"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        Console.info(
            f"Using GPU: {torch.cuda.get_device_name(0)} for blazingly fast training"
        )
        device = torch.device("cuda")
    else:
        Console.warning("Using CPU for training, this may be slow")
        device = torch.device("cpu")

    Console.info("Starting pre-training phase")
    pipeline = Pipeline(encoder, run_id)

    pipeline = pipeline >> MaskedLanguageModelPretrainingStage(
        run_id, device
    ).with_sources(
        train_source=FastaSliceSource(
            directory=Path("temp/data/external/sequences/nucleotides"),
            metadata=Path("temp/data/processed/pre_training.parquet"),
            key_column="genomic_nucleotide_accession.version",
            start_column="start_position_on_the_genomic_accession",
            end_column="end_position_on_the_genomic_accession",
            orientation_column="orientation",
            sequence_column="dna_sequence",
        ),
        valid_source=FastaSliceSource(
            directory=Path("temp/data/external/sequences/nucleotides"),
            metadata=Path("temp/data/processed/fine_tuning.valid.parquet"),
            key_column="genomic_nucleotide_accession.version",
            start_column="start_position_on_the_genomic_accession",
            end_column="end_position_on_the_genomic_accession",
            orientation_column="orientation",
            sequence_column="dna_sequence",
        ),
    )

    pretrained_encoder = pipeline.run()

    Console.info("Starting fine-tuning phase")

    pipeline = pipeline >> TripletMetricFinetuningStage(run_id, device).with_sources(
        train_source=(
            TabularSequenceSource(
                source=Path("temp/data/processed/card_amr.train.parquet")
            )
            + FastaSliceSource(
                directory=Path("temp/data/external/sequences/nucleotides"),
                metadata=Path("temp/data/processed/fine_tuning.train.parquet"),
                key_column="genomic_nucleotide_accession.version",
                start_column="start_position_on_the_genomic_accession",
                end_column="end_position_on_the_genomic_accession",
                orientation_column="orientation",
                sequence_column="dna_sequence",
            )
        ),
        valid_source=(
            TabularSequenceSource(
                source=Path("temp/data/processed/card_amr.valid.parquet")
            )
            + FastaSliceSource(
                directory=Path("temp/data/external/sequences/nucleotides"),
                metadata=Path("temp/data/processed/fine_tuning.valid.parquet"),
                key_column="genomic_nucleotide_accession.version",
                start_column="start_position_on_the_genomic_accession",
                end_column="end_position_on_the_genomic_accession",
                orientation_column="orientation",
                sequence_column="dna_sequence",
            )
        ),
    )

    pipeline = pipeline >> MultiLabelClassificationFinetuningStage(
        run_id, device
    ).with_sources(
        train_source=(
            TabularSequenceSource(
                source=Path("temp/data/processed/card_amr.train.parquet")
            )
            + FastaSliceSource(
                directory=Path("temp/data/external/sequences/nucleotides"),
                metadata=Path("temp/data/processed/fine_tuning.train.parquet"),
                key_column="genomic_nucleotide_accession.version",
                start_column="start_position_on_the_genomic_accession",
                end_column="end_position_on_the_genomic_accession",
                orientation_column="orientation",
                sequence_column="dna_sequence",
            )
        ),
        valid_source=(
            TabularSequenceSource(
                source=Path("temp/data/processed/card_amr.valid.parquet")
            )
            + FastaSliceSource(
                directory=Path("temp/data/external/sequences/nucleotides"),
                metadata=Path("temp/data/processed/fine_tuning.valid.parquet"),
                key_column="genomic_nucleotide_accession.version",
                start_column="start_position_on_the_genomic_accession",
                end_column="end_position_on_the_genomic_accession",
                orientation_column="orientation",
                sequence_column="dna_sequence",
            )
        ),
    )

    finetuned_encoder = pipeline.run()

    save_model(
        model=pretrained_encoder,
        name="language_model_encoder",
        run_id=run_id,
    )

    save_model(
        model=finetuned_encoder,
        name="classification_model_encoder",
        run_id=run_id,
    )

    save_model(
        model=pipeline.last_completed_stage.model
        if pipeline.last_completed_stage
        else ClassificationModel(encoder=finetuned_encoder, number_of_classes=1),
        name="classification_model_full",
        run_id=run_id,
    )

    Console.info("Training complete")
    return


if __name__ == "__main__":
    main()
    exit(0)
