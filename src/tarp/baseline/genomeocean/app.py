from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
from xgboost import XGBClassifier

from tarp.data.datasets.classification.multilabel import MultiLabelClassificationDataset
from tarp.data.sources.sequence import (
    SequenceDataSource,
    TabularSequenceSource,
)
from tarp.model.backbone.pretrained.genomeocean import GenomeOceanEncoder
from tarp.preprocessing.tokenizers.pretrained.genomeocean import GenomeOceanTokenizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = GenomeOceanTokenizer()
    encoder = GenomeOceanEncoder().freeze().to(device)  # Freeze the encoder weights

    label_columns = cast(
        list[str],
        (pl.read_csv(Path("temp/data/cache/labels_reduced.csv")).to_series().to_list()),
    )
    label_columns.remove("non_amr")

    classification_training_dataset = MultiLabelClassificationDataset(
        source=cast(
            SequenceDataSource[dict[str, str | float]],
            TabularSequenceSource(
                source=Path(
                    "temp/data/processed/finetuning/fold_0/train/card_amr.parquet"
                ),
            ),
        ),
        tokenizer=tokenizer,
        sequence_column="dna_sequence",
        label_columns=label_columns,
        maximum_sequence_length=512,
    )

    classification_test_dataset = MultiLabelClassificationDataset(
        source=cast(
            SequenceDataSource[dict[str, str | float]],
            TabularSequenceSource(
                source=Path(
                    "temp/data/processed/finetuning/fold_0/test/card_amr.parquet"
                ),
            ),
        ),
        tokenizer=tokenizer,
        sequence_column="dna_sequence",
        label_columns=label_columns,
        maximum_sequence_length=512,
    )

    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for i in tqdm(
            range(len(classification_training_dataset)),
            desc="Extracting training embeddings",
        ):
            item = classification_training_dataset[i]
            sequence = item["sequence"].to(device)
            attention_mask = item["attention_mask"].to(device)

            embedding, _ = encoder(
                sequence.unsqueeze(0), attention_mask.unsqueeze(0), mode="pooled"
            )
            embeddings.append(embedding.detach().cpu().to(torch.float32).numpy())
            labels.append(item["labels"].detach().cpu().numpy())

    X_train = np.vstack(embeddings)
    Y_train = np.vstack(labels)

    print("Training XGBoost classifier...")

    classifier = XGBClassifier(
        n_estimators=200,
        max_depth=7,
        objective="binary:logistic",
        learning_rate=0.1,
    )

    print("Evaluating on test set...")

    embeddings = []
    labels = []
    with torch.no_grad():
        for i in tqdm(
            range(len(classification_test_dataset)),
            desc="Extracting test embeddings",
        ):
            item = classification_test_dataset[i]
            sequence = item["sequence"].to(device)
            attention_mask = item["attention_mask"].to(device)

            embedding, _ = encoder(
                sequence.unsqueeze(0), attention_mask.unsqueeze(0), mode="pooled"
            )
            embeddings.append(embedding.detach().cpu().to(torch.float32).numpy())
            labels.append(item["labels"].detach().cpu().numpy())

    X_test = np.vstack(embeddings)
    Y_test = np.vstack(labels)

    Y_pred = classifier.predict(X_test)

    print("Classification report:")
    print(str(classification_report(Y_test, Y_pred)))


if __name__ == "__main__":
    print("Running GenomeOcean baseline...")
    main()
