from collections.abc import Sequence
from pathlib import Path

import polars as pl
import torch
from torch import Tensor

from tarp.cli.logging import Console
from tarp.services.datasets import SequenceDataset
from tarp.services.datasets.classification.multilabel import (
    MultiLabelClassificationDataset,
)


class MultiLabelOfflineTripletDataset(SequenceDataset[dict[str, dict[str, Tensor]]]):
    """
    Dataset for generating triplets (anchor, positive, negative) from a multi-label classification dataset.

    - Positive: random sample sharing ≥1 label.
    - Negative: random sample sharing no labels.

    If no positive or negative is found, falls back gracefully.
    """

    def __init__(
        self,
        base_dataset: MultiLabelClassificationDataset,
        label_cache: Path = Path("temp/data/interim/labels_cache.parquet"),
    ):
        super().__init__(
            base_dataset.data_source,
            base_dataset.tokenizer,
            base_dataset.sequence_column,
            base_dataset.augmentation,
        )
        self.base_dataset = base_dataset
        self.data_source = base_dataset.data_source
        if self.data_source.height < 2:
            raise ValueError("Base dataset must contain at least two samples.")

        expected_columns = base_dataset.label_columns

        self.labels = None

        if label_cache.exists():
            Console.debug(f"Checking for label cache at: {label_cache}")

            df = pl.read_parquet(label_cache)
            cached_columns = df.columns

            # Ensure column order and names match expected
            if (
                set(cached_columns) == set(expected_columns)
                and df.shape[0] == self.data_source.height
            ):
                # Reorder columns to match expected order
                df = df.select(expected_columns)
                self.labels = torch.tensor(df.to_numpy(), dtype=torch.float32)
                Console.info(
                    f"Loaded labels from cache (aligned to label_columns): {label_cache}"
                )
            else:
                Console.warning(
                    f"Label cache mismatch — columns or size differ. "
                    f"Expected shape ({self.data_source.height}, {len(expected_columns)}), "
                    f"found shape {df.shape}, column difference: {list(set(cached_columns) - set(expected_columns))}"
                )

        # If cache missing or mismatched, recompute and save
        if self.labels is None:
            Console.warning("Computing labels from base dataset. This may take a while")
            self.labels = torch.stack(
                [self.base_dataset[i]["labels"] for i in range(len(self.base_dataset))]
            )

            # Convert to Polars DataFrame for saving
            pl.DataFrame(self.labels.numpy(), schema=expected_columns).write_parquet(
                label_cache
            )
            Console.info(f"Saved labels to cache: {label_cache}")

        with torch.no_grad():
            self.overlap_matrix = (self.labels @ self.labels.T) > 0
            self.no_overlap_matrix = ~self.overlap_matrix

            diagonal = torch.eye(len(self.base_dataset), dtype=torch.bool)
            self.overlap_matrix = self.overlap_matrix & ~diagonal
            self.no_overlap_matrix = self.no_overlap_matrix & ~diagonal

    def process_row(self, index: int, row: dict) -> dict[str, dict[str, Tensor]]:
        positive_mask = self.overlap_matrix[index]
        negative_mask = self.no_overlap_matrix[index]

        # Random positive (shares at least one label)
        positive_indices = torch.where(positive_mask)[0]
        if len(positive_indices) > 0:
            positive_index = int(
                positive_indices[torch.randint(len(positive_indices), (1,))]
            )
        else:
            positive_index = index

        # Random negative (shares no labels)
        negative_indices = torch.where(negative_mask)[0]
        if len(negative_indices) > 0:
            negative_index = int(
                negative_indices[torch.randint(len(negative_indices), (1,))]
            )
        else:
            negative_index = index

        return {
            "anchor": self.base_dataset.process_row(index, row),
            "positive": self.base_dataset[positive_index],
            "negative": self.base_dataset[negative_index],
        }

    def collate_function(
        self, batch: Sequence[dict[str, dict[str, Tensor]]]
    ) -> dict[str, dict[str, Tensor]]:
        collated = {"anchor": {}, "positive": {}, "negative": {}}
        for key in ["anchor", "positive", "negative"]:
            # Use the collate_function from the base dataset to pad sequences and attention masks and stack labels
            collated[key] = self.base_dataset.collate_function(
                [item[key] for item in batch]
            )
        return collated
