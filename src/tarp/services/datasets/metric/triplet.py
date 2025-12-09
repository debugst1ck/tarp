from pathlib import Path

import polars as pl
import torch
from torch import Tensor

from tarp.cli.logging import Console
from tarp.services.datasets import SequenceDataset
from tarp.services.datasets.classification.multilabel import (
    MultiLabelClassificationDataset,
)


class MultiLabelOfflineTripletDataset(SequenceDataset):
    """
    Dataset for generating triplets (anchor, positive, negative) from a multi-label classification dataset.

    - Hard positive: closest sample that shares at least one label with the anchor.
    - Hard negative: farthest sample that shares no labels with the anchor.

    If no positive or negative is found, falls back gracefully.
    """

    def __init__(
        self, base_dataset: MultiLabelClassificationDataset, label_cache: Path = None
    ):
        self.base_dataset = base_dataset
        self.data_source = base_dataset.data_source
        if self.data_source.height < 2:
            raise ValueError("Base dataset must contain at least two samples.")

        expected_columns = base_dataset.label_columns

        self.labels = None

        if label_cache and label_cache.exists():
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
            df = pl.DataFrame(
                self.labels.numpy(), schema=expected_columns
            ).write_parquet(
                label_cache
                if label_cache
                else Path("temp/data/interim/labels_cache.parquet")
            )
            Console.info(f"Saved labels to cache: {label_cache}")

        with torch.no_grad():
            self.overlap_matrix = (self.labels @ self.labels.T) > 0
            self.distance_matrix = torch.cdist(self.labels, self.labels, p=2)
            self.no_overlap_matrix = ~self.overlap_matrix

            # Mask out self-distances for positive mining
            diagonal = torch.eye(len(self.base_dataset), dtype=torch.bool)
            self.overlap_matrix = self.overlap_matrix & ~diagonal
            self.distance_matrix = self.distance_matrix.masked_fill(
                diagonal, float("inf")
            )

    def process_row(self, index: int, row: dict) -> dict[str, dict[str, Tensor]]:
        positive_mask = self.overlap_matrix[index]
        negative_mask = self.no_overlap_matrix[index]
        distances = self.distance_matrix[index]

        # Hard positive: closest sample that shares at least one label
        if positive_mask.any():
            positive_index = (
                distances.masked_fill(~positive_mask, float("inf")).argmin().item()
            )
        else:
            positive_index = index  # Fallback to self if no positive found

        # Hard negative: farthest sample that shares no labels
        if negative_mask.any():
            negative_index = (
                distances.masked_fill(~negative_mask, float("-inf")).argmax().item()
            )
        else:
            negative_index = index  # Fallback to self if no negative found

        return {
            "anchor": self.base_dataset.process_row(index, row),
            "positive": self.base_dataset[positive_index],
            "negative": self.base_dataset[negative_index],
        }
