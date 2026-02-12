from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl
import torch
from Bio import SeqIO
from torch import Tensor

from tarp.cli.logging import Console


# Mru cache could be used for caching sequences if needed
class SequenceDataSource(ABC):
    """
    Encapsulate file interactions for sequence datasets.
    """

    @property
    def height(self) -> int:
        """
        Get the number of rows in the data source.

        :return int: The height of the sequence.
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, index: int) -> dict:
        """
        Retrieve a single row from the data source.

        :param int index: The index of the row to retrieve.
        :return dict: A dictionary representation of the row.
        """
        raise NotImplementedError

    def batch(self, indices: Sequence[int]) -> Sequence[dict]:
        """
        Retrieve multiple rows from the data source.

        :param list[int] indices: The indices of the rows to retrieve.
        :return list[dict]: A list of dictionary representations of the rows.
        """
        return [self.retrieve(i) for i in indices]

    def __add__(self, other: "SequenceDataSource") -> "CombinationSource":
        """
        Combine two data sources into one.

        :param SequenceDataSource other: The other data source to combine with.
        :return CombinationSource: A new CombinationSource instance.
        """
        # If both are CombinationSource, flatten their sources
        if isinstance(self, CombinationSource) and isinstance(other, CombinationSource):
            return CombinationSource(self.sources + other.sources)
        # If only self is CombinationSource, append other
        elif isinstance(self, CombinationSource):
            return CombinationSource(self.sources + [other])
        # If only other is CombinationSource, prepend self
        elif isinstance(other, CombinationSource):
            return CombinationSource([self] + other.sources)
        # Otherwise, create a new CombinationSource with both
        else:
            return CombinationSource([self, other])


class TabularSequenceSource(SequenceDataSource):
    """
    Reads from a tabular data source (e.g., CSV, Excel, Parquet). Stores in a Polars DataFrame.
    """

    def __init__(self, source: Path):
        self.source = source
        self.dataframe: Optional[pl.DataFrame] = None
        self._read_source()

    def _read_source(self) -> None:
        if self.source.suffix == ".csv":
            self.dataframe = pl.read_csv(self.source)
        elif self.source.suffix == ".xlsx":
            self.dataframe = pl.read_excel(self.source)
        elif self.source.suffix == ".parquet":
            self.dataframe = pl.read_parquet(self.source)
        else:
            raise ValueError(f"Unsupported file type: {self.source.suffix}")

        # Add index column if not present
        if "index" not in self.dataframe.columns:
            self.dataframe = self.dataframe.with_row_index("index")

    @property
    def height(self) -> int:
        if self.dataframe is not None:
            return self.dataframe.height
        return 0

    def retrieve(self, index: int) -> dict:
        if self.dataframe is not None:
            return self.dataframe.row(index, named=True)
        return {}

    def batch(self, indices: Sequence[int]) -> Sequence[dict]:
        if self.dataframe is not None:
            return self.dataframe.filter(pl.col("index").is_in(indices)).rows(
                named=True
            )
        return []


class FastaDirectorySource(SequenceDataSource):
    """
    Reads from a directory of FASTA files. Metadata is stored as a Tabular source.
    """

    def __init__(
        self,
        directory: Path,
        metadata: Path,
        key_column: str,
        sequence_column: str = "sequence",
    ):
        self.directory = directory
        self.metadata = metadata
        self.key_column = key_column
        self.sequence_column = sequence_column

        self.dataframe: Optional[pl.DataFrame] = None
        self._read_source()

    def _read_source(self) -> None:
        if self.metadata.suffix == ".csv":
            self.dataframe = pl.read_csv(self.metadata)
        elif self.metadata.suffix == ".xlsx":
            self.dataframe = pl.read_excel(self.metadata)
        elif self.metadata.suffix == ".parquet":
            self.dataframe = pl.read_parquet(self.metadata)
        else:
            raise ValueError(f"Unsupported file type: {self.metadata.suffix}")

    @property
    def height(self) -> int:
        if self.dataframe is not None:
            return self.dataframe.height
        return 0

    def retrieve(self, index: int) -> dict:
        # Retrieve the key column which correspond to Fasta file name
        if self.dataframe is None:
            return {}

        key = self.dataframe[index][self.key_column]
        fasta_path = self.directory / f"{key}.fasta"

        if not fasta_path.exists():
            return {}

        with open(fasta_path, "r") as handle:
            row = self.dataframe[index].to_dict(as_series=False)
            row: dict[str, Any] = dict(row)
            row[self.sequence_column] = "".join(
                str(record.seq) for record in SeqIO.parse(handle, "fasta")
            )
        return row


class CombinationSource(SequenceDataSource):
    """
    Combines multiple data sources into one.
    """

    def __init__(self, sources: list[SequenceDataSource]):
        self.sources = sources
        self._cumulative_heights = self._compute_cumulative_heights()

    def _compute_cumulative_heights(self) -> Tensor:
        heights = torch.tensor(
            [0] + [source.height for source in self.sources], dtype=torch.long
        )
        return torch.cumsum(heights, dim=0)

    @property
    def height(self) -> int:
        return sum(source.height for source in self.sources)

    def retrieve(self, index: int) -> dict:
        source_index = int(
            torch.searchsorted(self._cumulative_heights, index, right=True) - 1
        )
        local_index = index - int(self._cumulative_heights[source_index])
        return self.sources[source_index].retrieve(local_index)

    def batch(self, indices: Sequence[int]) -> Sequence[dict]:
        # We need to preserve the order of indices
        indices_t = torch.as_tensor(indices, dtype=torch.long)
        # Bucketize indices by source
        source_indices = (
            torch.bucketize(indices_t, self._cumulative_heights, right=True) - 1
        )
        # Compute local indices within each source
        local_indices = indices_t - self._cumulative_heights[source_indices]
        # Prepare output buffer
        results: list[dict] = [{} for _ in range(len(indices))]
        for source_index in torch.unique(source_indices):
            mask = source_indices == source_index
            positions = torch.nonzero(mask, as_tuple=False).squeeze(1)
            source_local_indices = local_indices[mask].tolist()
            batch_results = self.sources[int(source_index.item())].batch(
                source_local_indices
            )
            # Assign batch results to original positions
            for global_position, result in zip(positions.tolist(), batch_results):
                results[global_position] = result
        return results


class InMemorySequenceSource(SequenceDataSource):
    """
    Stores sequences in memory for fast access.
    """

    def __init__(self, data: pl.DataFrame):
        self.dataframe = data

    @property
    def height(self) -> int:
        return self.dataframe.height

    def retrieve(self, index: int) -> dict:
        return self.dataframe.row(index, named=True)

    def batch(self, indices: Sequence[int]) -> Sequence[dict]:
        return self.dataframe.filter(pl.col("index").is_in(indices)).rows(named=True)


class FastaSliceSource(SequenceDataSource):
    """
    Reads a slice of sequences from FASTA files. Uses LRU caching for efficiency.
    """

    def __init__(
        self,
        directory: Path,
        metadata: Path,
        key_column: str,
        start_column: str,
        end_column: str,
        sequence_column: str = "sequence",
    ):
        self.directory = directory
        self.metadata = metadata
        self.key_column = key_column
        self.start_column: str | None = start_column
        self.end_column: str | None = end_column
        self.sequence_column = sequence_column

        self.df = (
            pl.read_parquet(metadata)
            if metadata.suffix == ".parquet"
            else pl.read_csv(metadata)
        )

        if self.key_column not in self.df.columns:
            raise ValueError(
                f"Key column {self.key_column} not found in {self.df.columns}."
            )

        if self.start_column not in self.df.columns:
            Console.warning(f"Start column {self.start_column} not found in metadata.")
            self.start_column = None  # Invalidate if not found

        if self.end_column not in self.df.columns:
            Console.warning(f"End column {self.end_column} not found in metadata.")
            self.end_column = None  # Invalidate if not found

        self._fasta_map = {p.stem: p for p in self.directory.glob("*.fasta")}

    @lru_cache(maxsize=1024)
    def _load_sequence(self, key: str) -> str:
        return self._load_sequence_uncached(key)

    def _load_sequence_uncached(self, key: str) -> str:
        """
        Load a full genome sequence from FASTA.

        :param str key: The key corresponding to the FASTA file.
        :return str: The full genome sequence as a string.
        """
        fasta_path = self._fasta_map.get(key)
        if not fasta_path:
            raise FileNotFoundError(f"No FASTA found for {key}")

        with open(fasta_path) as handle:
            rec = next(SeqIO.parse(handle, "fasta"))
            return str(rec.seq)

    @property
    def height(self) -> int:
        return self.df.height

    def retrieve(self, index: int) -> dict:
        row = self.df.row(index, named=True)
        key = row[self.key_column]
        start = row.get(self.start_column) if self.start_column else None
        end = row.get(self.end_column) if self.end_column else None

        if key not in self._fasta_map:
            raise FileNotFoundError(f"No FASTA found for {key}")

        full_sequence = self._load_sequence(key)

        # Cast to str for compatibility with slicing operations
        full_sequence = str(full_sequence)

        if start is None or end is None:
            sequence = full_sequence
        else:
            sequence = full_sequence[start:end]
        row[self.sequence_column] = sequence
        return row

    def batch(self, indices: Sequence[int]) -> Sequence[dict]:
        # Torch based batch implementation for efficiency
        subset = self.df[indices]
        keys = subset[self.key_column].to_numpy()
        starts = subset[self.start_column].to_numpy() if self.start_column else None
        ends = subset[self.end_column].to_numpy() if self.end_column else None

        # Prepare output buffer
        results = [{} for _ in range(len(indices))]

        # Group by keys to minimize file reads, use torch.unique
        unique_keys, inverse_indices = np.unique(keys, return_inverse=True)

        for global_index, key in enumerate(unique_keys):
            if key not in self._fasta_map:
                raise FileNotFoundError(
                    f"No FASTA found for {key} in {self.directory.as_posix()}"
                )

            # Mask for current key
            mask = inverse_indices == global_index
            positions = np.nonzero(mask)[0]

            full_sequence = self._load_sequence(key)
            if starts is None or ends is None:
                # No slicing needed, assign full sequence
                for position in positions.tolist():
                    row = subset.row(position, named=True)
                    row[self.sequence_column] = full_sequence
                    results[position] = row
            else:
                group_starts = starts[positions]
                group_ends = ends[positions]

                for i, position in enumerate(positions.tolist()):
                    start = int(group_starts[i])
                    end = int(group_ends[i])
                    row = subset.row(position, named=True)
                    sequence = full_sequence[start:end]
                    row[self.sequence_column] = sequence
                    results[position] = row
        return results
