# ESM1b

from pathlib import Path

import numpy as np
import polars as pl
import torch
from numpy.typing import NDArray
from sklearn.multioutput import MultiOutputClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

from tarp.cli.logging import Console
from tarp.model.backbone.pretrained.dnabert2 import FrozenDnabert2Encoder
from tarp.services.datasets.classification.multilabel import (
    MultiLabelClassificationDataset,
)
from tarp.services.datasources.sequence import (
    FastaSliceSource,
    TabularSequenceSource,
)
from tarp.services.evaluation.classification.multilabel import MultiLabelMetrics
from tarp.services.preprocessing.augmentation import (
    CompositeAugmentation,
)
from tarp.services.preprocessing.augmentation.protein import (
    InsertionDeletion,
    RandomMutation,
)
from tarp.services.tokenizers.pretrained.dnabert2 import Dnabert2Tokenizer


def main():
    classification_head = MultiOutputClassifier(
        XGBClassifier(
            learning_rate=0.1,
            max_depth=7,
            n_estimators=200,
            objective="binary:logistic",
            verbosity=1,  # show progress for each boosting round
        )
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_encoder = FrozenDnabert2Encoder(768).to(device)
    tokenizer = Dnabert2Tokenizer()
    label_columns = (
        pl.read_csv(Path("temp/data/cache/labels.csv")).to_series().to_list()
    )

    Console.info("Model and classification head initialized")

    multilabel_classification_train = MultiLabelClassificationDataset(
        FastaSliceSource(
            directory=Path("temp/data/external/sequences/proteins"),
            metadata=Path("temp/data/processed/fine_tuning.train.parquet"),
            key_column="protein_accession.version",
            start_column="?",
            end_column="?",
            sequence_column="protein_sequence",
        )
        + TabularSequenceSource(
            source=Path("temp/data/processed/card_amr.train.parquet"),
        ),
        tokenizer=tokenizer,
        sequence_column="protein_sequence",
        label_columns=label_columns,
        maximum_sequence_length=200,
        augmentation=CompositeAugmentation(
            [
                RandomMutation(),
                InsertionDeletion(),
            ]
        ),
    )

    multilabel_classification_valid = MultiLabelClassificationDataset(
        (
            TabularSequenceSource(
                source=Path("temp/data/processed/card_amr.valid.parquet"),
            )
            + FastaSliceSource(
                directory=Path("temp/data/external/sequences/proteins"),
                metadata=Path("temp/data/processed/fine_tuning.valid.parquet"),
                key_column="protein_accession.version",
                start_column="?",
                end_column="?",
                sequence_column="protein_sequence",
            )
        ),
        tokenizer=tokenizer,
        sequence_column="protein_sequence",
        label_columns=label_columns,
        maximum_sequence_length=200,
    )

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        multilabel_classification_train,
        batch_size=16,
        shuffle=True,
        num_workers=4,
    )

    valid_loader = DataLoader(
        multilabel_classification_valid,
        batch_size=16,
        shuffle=False,
        num_workers=4,
    )

    Console.info("Datasets initialized")

    x_embeddings: list[torch.Tensor] = []
    y_labels: list[torch.Tensor] = []

    # Encode training set
    for batch in tqdm(train_loader, desc="Encoding training"):
        input_ids = batch["sequence"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        with torch.no_grad():
            pooled = pretrained_encoder.encode(
                input_ids,
                attention_mask=attention_mask,
                return_sequence=False,
            )  # (B, hidden)

        # Keep as torch tensors
        x_embeddings.append(pooled.cpu())
        y_labels.append(labels.cpu())

    # Concatenate all tensors along batch dimension
    stacked_x = torch.cat(x_embeddings, dim=0)  # shape: (num_samples, hidden)
    stacked_y = torch.cat(y_labels, dim=0)  # shape: (num_samples, num_labels)

    # Optional: check zero-only label columns
    zero_only_cols = torch.where(stacked_y.sum(dim=0) == 0)[0]
    if len(zero_only_cols) > 0:
        Console.warning(
            f"Found {len(zero_only_cols)} label columns with only zeros. Removing them from training."
        )

    # Convert to numpy
    stacked_x_t = stacked_x.cpu().numpy()
    stacked_y_t = stacked_y.cpu().numpy()

    # Zero only label cols
    zero_only_cols = np.where(stacked_y_t.sum(axis=0) == 0)[0]
    if len(zero_only_cols) > 0:
        Console.warning(
            f"Found {len(zero_only_cols)} label columns with only zeros. Removing them from training."
        )

    Console.info("Starting training classification head")
    classification_head.fit(stacked_x_t, stacked_y_t)

    Console.info("Classification head training complete")

    # Validation
    x_val_embeddings: list[NDArray] = []
    y_val_labels: list[NDArray] = []

    for batch in tqdm(valid_loader, desc="Encoding validation"):
        input_ids = batch["sequence"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        with torch.no_grad():
            pooled = pretrained_encoder.encode(
                input_ids,
                attention_mask=attention_mask,
                return_sequence=False,
            )  # (B, hidden)

        x_val_embeddings.append(pooled.cpu().numpy())
        y_val_labels.append(labels.numpy())

    # np.vstack to create 2D arrays
    stacked_x_val = np.concatenate(x_val_embeddings, axis=0)
    stacked_y_val = np.concatenate(y_val_labels, axis=0)

    Console.info("Starting validation")
    val_predictions = classification_head.predict_proba(stacked_x_val)

    val_predictions_stacked = np.column_stack([p[:, 1] for p in val_predictions])

    metrics = MultiLabelMetrics().compute(
        torch.as_tensor(val_predictions_stacked),
        torch.as_tensor(stacked_y_val),
    )
    for metric_name, metric_value in metrics.items():
        Console.info(f"Validation {metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
