from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
import polars as pl
import torch
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from torch import Tensor
from torch.utils.data import DataLoader

from odyssey import AcceleratedRuntime, Orchestrator, Plugin, State
from tarp.data.datasets.classification.multilabel import MultiLabelClassificationDataset
from tarp.data.sources.sequence import TabularSequenceSource
from tarp.model.backbone.core import Encoder
from tarp.model.backbone.pretrained.genomeocean import GenomeOceanEncoder
from tarp.preprocessing.tokenizers.pretrained.genomeocean import GenomeOceanTokenizer
from tarp.typed.batch import ClassificationBatch


@dataclass(slots=True)
class ExtractionResult:
    loss: Tensor
    embedding: Tensor
    labels: Tensor


class EmbeddingExtractionObjective:
    def preprocess(
        self, batch: ClassificationBatch, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Move tensors to hardware accelerator
        seq = batch["sequence"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        return seq, mask, labels

    @torch.compile(mode="max-autotune-no-cudagraphs")
    def compute(
        self, model: Encoder, seq: Tensor, mask: Tensor, labels: Tensor
    ) -> ExtractionResult:
        # Extract features from frozen ESM backbone
        embedding, _ = model(seq, mask, mode="pooled")

        return ExtractionResult(
            loss=torch.tensor(0.0, device=seq.device),
            embedding=embedding.detach(),
            labels=labels,
        )

    def forward_pass(
        self, model: Encoder, batch: ClassificationBatch, device: torch.device
    ) -> ExtractionResult:
        seq, mask, labels = self.preprocess(batch, device)
        return self.compute(model, seq, mask, labels)


class EmbeddingAccumulatorPlugin(Plugin[ExtractionResult]):
    def __init__(self) -> None:
        self.embeddings: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []

    @override
    def on_epoch_begin(self, state: State, is_training: bool) -> None:
        self.embeddings.clear()
        self.labels.clear()

    @override
    def on_batch_end(
        self, state: State, result: ExtractionResult, is_training: bool
    ) -> None:
        # Move to host memory and cast to numpy arrays
        self.embeddings.append(result.embedding.cpu().numpy())
        self.labels.append(result.labels.cpu().numpy())

    def get_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        return np.vstack(self.embeddings), np.vstack(self.labels)


def main():
    tokenizer = GenomeOceanTokenizer()
    encoder = GenomeOceanEncoder().freeze()  # Freeze the encoder weights

    label_columns = [
        c
        for c in pl.read_csv("temp/data/cache/labels_reduced.csv").to_series().to_list()
        if c != "non_amr"
    ]

    training_dataset = MultiLabelClassificationDataset(
        source=TabularSequenceSource(
            source=Path("temp/data/processed/finetuning/fold_0/train/card_amr.parquet"),
        ),
        tokenizer=tokenizer,
        sequence_column="dna_sequence",
        label_columns=label_columns,
        maximum_sequence_length=1024,
    )

    test_dataset = MultiLabelClassificationDataset(
        source=TabularSequenceSource(
            source=Path("temp/data/processed/finetuning/fold_0/test/card_amr.parquet"),
        ),
        tokenizer=tokenizer,
        sequence_column="dna_sequence",
        label_columns=label_columns,
        maximum_sequence_length=1024,
    )

    train_loader = DataLoader(
        training_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=training_dataset.collate,
        pin_memory=True,
        num_workers=4,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=training_dataset.collate,
        pin_memory=True,
        num_workers=4,
    )

    accumulator = EmbeddingAccumulatorPlugin()
    orchestrator = Orchestrator(
        engine=AcceleratedRuntime(model=encoder),
        objective=EmbeddingExtractionObjective(),
        optimizers=(),  # Null iterable passed: no optimization steps occur
        plugins=(accumulator,),
    )

    state_train = State()
    _ = orchestrator.run(train_loader, state_train, is_training=False)
    X_train, Y_train = accumulator.get_dataset()

    print(f"Training dataset shape: {X_train.shape}, {Y_train.shape}")

    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1).fit(
        X_train, Y_train
    )

    state_test = State()
    _ = orchestrator.run(test_loader, state_test, is_training=False)
    X_test, Y_test = accumulator.get_dataset()

    print(f"Test dataset shape: {X_test.shape}, {Y_test.shape}")

    print("Generating classification report for test dataset...")

    Y_pred = knn.predict(X_test)

    print(
        classification_report(
            Y_test, Y_pred, zero_division=0.0, target_names=label_columns
        )
    )


if __name__ == "__main__":
    main()
