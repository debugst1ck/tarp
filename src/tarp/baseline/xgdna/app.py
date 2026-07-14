from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import torch
from sklearn.metrics import classification_report
from torch import nn
from tqdm.auto import tqdm
from xgboost import XGBClassifier

from tarp.data.datasets.classification.multilabel import MultiLabelClassificationDataset
from tarp.data.sources.sequence import (
    SequenceDataSource,
    TabularSequenceSource,
)
from tarp.model.backbone.untrained.transformer import TransformerEncoder
from tarp.model.layers.positional.core import (
    HeterogeneousTransformativePositionalEncoding,
)
from tarp.model.layers.positional.rotational import (
    CachedIntegerRotaryPositionalEncoding,
)
from tarp.model.tasks.language import LanguageModel
from tarp.preprocessing.tokenizers.atomic.monomer import NucleotideTokenizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dna_tokenizer = NucleotideTokenizer()

    embedding_dimension = 576
    dna_embedding = nn.Embedding(
        num_embeddings=dna_tokenizer.vocabulary_size,
        embedding_dim=embedding_dimension,
        padding_idx=dna_tokenizer.pad_token_id,
    )

    encoder = TransformerEncoder(
        model_dimension=embedding_dimension,
        number_of_layers=embedding_dimension // 64,
        number_of_heads=embedding_dimension // 64,
        feed_forward_dimension=(embedding_dimension * 8) // 3,
        positional_encoder=HeterogeneousTransformativePositionalEncoding(
            CachedIntegerRotaryPositionalEncoding(64, base=10_000),
            CachedIntegerRotaryPositionalEncoding(64, base=10_000),
        ),
        dropout=0.1,
    )

    language_model = LanguageModel(
        encoder=encoder,
        embedding=dna_embedding,
        vocabulary_size=dna_tokenizer.vocabulary_size,
    ).to(device)

    model_save_dir = Path("temp/models/")
    # torch.save(language_model.state_dict(), model_save_dir / "language_model.pth")
    # Load the model weights
    language_model.load_state_dict(
        torch.load(model_save_dir / "language_model.pth", map_location=device)
    )

    label_columns = (
        pl.read_csv(Path("temp/data/cache/labels_reduced.csv")).to_series().to_list()
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
        tokenizer=dna_tokenizer,
        sequence_column="dna_sequence",
        label_columns=label_columns,
        maximum_sequence_length=1024,
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
        tokenizer=dna_tokenizer,
        sequence_column="dna_sequence",
        label_columns=label_columns,
        maximum_sequence_length=1024,
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

            sequence_embedding = dna_embedding(sequence.unsqueeze(0))
            embedding, _ = encoder(
                sequence_embedding, attention_mask.unsqueeze(0), mode="pooled"
            )

            embeddings.append(embedding.detach().cpu().numpy())
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

    classifier.fit(X_train, Y_train)

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

            sequence_embedding = dna_embedding(sequence.unsqueeze(0))
            embedding, _ = encoder(
                sequence_embedding, attention_mask.unsqueeze(0), mode="pooled"
            )
            embeddings.append(embedding.detach().cpu().numpy())
            labels.append(item["labels"].detach().cpu().numpy())

    X_test = np.vstack(embeddings)
    Y_test = np.vstack(labels)

    Y_pred = classifier.predict(X_test)

    print("Classification report:")
    print(classification_report(Y_test, Y_pred))


if __name__ == "__main__":
    print("Running xgdna baseline...")
    main()
