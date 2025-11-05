import datetime
import os
import torch
from pathlib import Path

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR

from torch.utils.data import Subset

import plotly.express as px
import polars as pl

from tarp.model.backbone.untrained.hyena import HyenaEncoder
from tarp.model.backbone.untrained.transformer import TransformerEncoder

from tarp.model.finetuning.language import LanguageModel
from tarp.model.finetuning.metric.triplet import TripletMetricModel
from tarp.model.finetuning.classification import ClassificationModel

from tarp.services.utilities.seed import establish_random_seed
from tarp.cli.logging import Console
from tarp.services.tokenizers.pretrained.dnabert import Dnabert2Tokenizer
from tarp.services.datasource.sequence import (
    TabularSequenceSource,
    FastaSliceSource,
)
from tarp.services.datasets.classification.multilabel import (
    MultiLabelClassificationDataset,
)
from tarp.services.datasets.metric.triplet import MultiLabelOfflineTripletDataset
from tarp.services.training.trainer.classification.multilabel import (
    MultiLabelClassificationTrainer,
)
from tarp.services.evaluation.losses.multilabel import AsymmetricFocalLoss
from tarp.services.training.trainer.metric.triplet import OfflineTripletMetricTrainer
from tarp.services.preprocessing.augmentation import (
    CombinationTechnique,
    RandomMutation,
    InsertionDeletion,
    ReverseComplement,
)

from tarp.services.datasets.language.masked import MaskedLanguageModelDataset

from tarp.config import HyenaConfig, TransformerConfig

from tarp.services.training.trainer.language.masked import MaskedLanguageModelTrainer

from sklearn.model_selection import train_test_split

import torch.multiprocessing as mp


def main() -> None:
    Console.info("App started")

    try:
        mp.set_start_method("spawn", force=True)
        Console.info("Multiprocessing start method set to 'spawn'")
        persistent_workers = False
    except RuntimeError:
        Console.warning("Multiprocessing start method was already set, skipping...")
        persistent_workers = True
        pass

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set seed for reproducibility
    SEED = establish_random_seed(69420)  # FuNnY NuMbEr :D
    Console.info(f"Random seed set to {SEED}")

    label_columns = (
        pl.read_csv(Path("temp/data/cache/labels.csv")).to_series().to_list()
    )

    # Create dataset
    multilabel_classification_train = MultiLabelClassificationDataset(
        (
            TabularSequenceSource(
                source=Path("temp/data/processed/card_amr.train.parquet")
            )
            + FastaSliceSource(
                directory=Path("temp/data/external/sequences"),
                metadata=Path("temp/data/processed/fine_tuning.train.parquet"),
                key_column="genomic_nucleotide_accession.version",
                start_column="start_position_on_the_genomic_accession",
                end_column="end_position_on_the_genomic_accession",
                orientation_column="orientation",
            )
        ),
        Dnabert2Tokenizer(),
        sequence_column="sequence",
        label_columns=label_columns,
        maximum_sequence_length=512,
        augmentation=CombinationTechnique(
            [
                RandomMutation(),
                InsertionDeletion(),
                ReverseComplement(0.5),
            ]
        ),
    )

    multilabel_classification_valid = MultiLabelClassificationDataset(
        (
            TabularSequenceSource(
                source=Path("temp/data/processed/card_amr.valid.parquet")
            )
            + FastaSliceSource(
                directory=Path("temp/data/external/sequences"),
                metadata=Path("temp/data/processed/fine_tuning.valid.parquet"),
                key_column="genomic_nucleotide_accession.version",
                start_column="start_position_on_the_genomic_accession",
                end_column="end_position_on_the_genomic_accession",
                orientation_column="orientation",
            )
        ),
        Dnabert2Tokenizer(),
        sequence_column="sequence",
        label_columns=label_columns,
        maximum_sequence_length=512,
    )

    masked_language_dataset_train = MaskedLanguageModelDataset(
        data_source=(
            FastaSliceSource(
                directory=Path("temp/data/external/sequences"),
                metadata=Path("temp/data/processed/pre_training.parquet"),
                key_column="genomic_nucleotide_accession.version",
                start_column="start_position_on_the_genomic_accession",
                end_column="end_position_on_the_genomic_accession",
                orientation_column="orientation",
            )
        ),
        tokenizer=Dnabert2Tokenizer(),
        sequence_column="sequence",
        maximum_sequence_length=512,
        masking_probability=0.15,
        augmentation=CombinationTechnique(
            [
                RandomMutation(),
                InsertionDeletion(),
                ReverseComplement(0.5),
            ]
        ),
    )

    masked_language_dataset_valid = MaskedLanguageModelDataset(
        data_source=(
            FastaSliceSource(
                directory=Path("temp/data/external/sequences"),
                metadata=Path("temp/data/processed/fine_tuning.valid.parquet"),
                key_column="genomic_nucleotide_accession.version",
                start_column="start_position_on_the_genomic_accession",
                end_column="end_position_on_the_genomic_accession",
                orientation_column="orientation",
            )
        ),
        tokenizer=Dnabert2Tokenizer(),
        sequence_column="sequence",
        maximum_sequence_length=512,
        masking_probability=0.15,
    )

    triplet_dataset_train = MultiLabelOfflineTripletDataset(
        base_dataset=multilabel_classification_train,
        label_cache=Path("temp/data/cache/labels_cache_train.parquet"),
    )

    triplet_dataset_valid = MultiLabelOfflineTripletDataset(
        base_dataset=multilabel_classification_valid,
        label_cache=Path("temp/data/cache/labels_cache_valid.parquet"),
    )

    label_cache = pl.read_parquet(Path("temp/data/cache/labels_cache_train.parquet"))
    label_tensor = torch.tensor(label_cache.select(label_columns).to_numpy())

    class_counts = label_tensor.sum(dim=0)
    total_counts = label_tensor.size(0)
    class_weights = (total_counts - class_counts) / class_counts
    class_weights = (class_weights - class_weights.min()) / (
        class_weights.max() - class_weights.min()
    )
    class_weights = class_weights * (10.0 - 0.1) + 0.1  # Scale to [0.1, 10.0]

    # Display pos weights as a polars DataFrame
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

    encoder = TransformerEncoder(
        vocabulary_size=multilabel_classification_train.tokenizer.vocab_size,
        embedding_dimension=TransformerConfig.embedding_dimension,
        hidden_dimension=TransformerConfig.hidden_dimension,
        padding_id=multilabel_classification_train.tokenizer.pad_token_id,
        number_of_layers=TransformerConfig.number_of_layers,
        number_of_heads=TransformerConfig.number_of_heads,
        dropout=TransformerConfig.dropout,
    )

    # encoder = HyenaEncoder(
    #     vocabulary_size=multi_label_classification_dataset.tokenizer.vocab_size,
    #     embedding_dimension=HyenaConfig.embedding_dimension,
    #     hidden_dimension=HyenaConfig.hidden_dimension,
    #     padding_id=multi_label_classification_dataset.tokenizer.pad_token_id,
    #     number_of_layers=HyenaConfig.number_of_layers,
    #     dropout=HyenaConfig.dropout,
    # )

    classification_model = ClassificationModel(
        encoder=encoder,
        number_of_classes=len(label_columns),
    )

    triplet_model = TripletMetricModel(
        encoder=encoder,
    )

    language_model = LanguageModel(
        encoder=encoder,
        vocabulary_size=masked_language_dataset_train.tokenizer.vocab_size,
    )

    Console.info(f"Training {classification_model.encoder.__class__.__name__} model")

    # Use torch compile to optimize the model
    Console.info("Compiling model with torch.compile")
    torch.compile(classification_model, mode="reduce-overhead")
    torch.compile(triplet_model, mode="reduce-overhead")
    torch.compile(language_model, mode="reduce-overhead")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        Console.info(
            f"Using GPU: {torch.cuda.get_device_name(0)} for blazingly fast training"
        )
        device = torch.device("cuda")
    else:
        Console.warning("Using CPU for training, this may be slow")
        device = torch.device("cpu")

    optimizer_language = AdamW(
        [
            {
                "params": language_model.encoder.parameters(),
                "lr": 1e-4,
                "weight_decay": 1e-2,
            },
            {
                "params": language_model.language_head.parameters(),
                "lr": 1e-3,
                "weight_decay": 1e-2,
            },
        ]
    )

    Console.info("Starting masked language model training")
    Console.debug(
        f"Vocabulary size: {masked_language_dataset_train.tokenizer.vocab_size}"
    )
    MaskedLanguageModelTrainer(
        model=language_model,
        train_dataset=masked_language_dataset_train,
        valid_dataset=masked_language_dataset_valid,
        optimizer=optimizer_language,
        scheduler=CosineAnnealingWarmRestarts(optimizer_language, T_0=5, T_mult=2),
        device=device,
        epochs=20,
        num_workers=4,
        batch_size=64,
        accumulation_steps=4,
        persistent_workers=persistent_workers,
        vocabulary_size=masked_language_dataset_train.tokenizer.vocab_size,
    ).fit()

    multi_triplet_optimizer = AdamW(
        triplet_model.parameters(), lr=2e-5, weight_decay=1e-3
    )
    OfflineTripletMetricTrainer(
        model=triplet_model,
        train_dataset=triplet_dataset_train,
        valid_dataset=triplet_dataset_valid,
        optimizer=multi_triplet_optimizer,
        scheduler=CosineAnnealingWarmRestarts(multi_triplet_optimizer, T_0=3, T_mult=2),
        device=device,
        epochs=6,
        num_workers=4,
        batch_size=64,
        accumulation_steps=4,
        persistent_workers=persistent_workers,
    ).fit()

    Console.info("Starting classification model training")
    optimizer_classification = AdamW(
        [
            {
                "params": classification_model.encoder.parameters(),
                "lr": 1e-5,
                "weight_decay": 1e-2,
            },
            {
                "params": classification_model.classification_head.parameters(),
                "lr": 1e-3,
                "weight_decay": 1e-2,
            },
        ]
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
        device=device,
        epochs=15,
        num_workers=4,
        batch_size=64,
        accumulation_steps=4,
        persistent_workers=persistent_workers,
    )
    trainer.fit()

    torch.save(
        classification_model.state_dict(),
        f"temp/checkpoints/{classification_model.encoder.__class__.__name__}_{run_id}.pt",
    )

    Console.info("Training complete")

    # Visualize training history
    history = trainer.context.state.history  # List of dicts
    history_df = pl.DataFrame(history)
    fig = px.line(history_df, y=history_df.columns, title="Training History")
    fig.write_html(
        f"temp/reports/{classification_model.encoder.__class__.__name__}_{run_id}.html"
    )


if __name__ == "__main__":
    main()
