from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import cast, final, override

import polars as pl
import torch
from Bio import SeqIO
from torch import Tensor

from tarp.typed.core import KnownT


class SequenceDataSource[RowT: Mapping[str, KnownT]](ABC):
    @property
    def height(self) -> int:
        """
        Get the number of rows in the data source.

        :return int: The height of the sequence.
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, index: int) -> RowT:
        """
        Retrieve a single row from the data source.

        :param int index: The index of the row to retrieve.
        :return dict: A dictionary representation of the row.
        """
        raise NotImplementedError

    def batch(self, indices: Sequence[int]) -> Sequence[RowT]:
        """
        Retrieve a batch of rows from the data source.

        :param Sequence[int] indices: A list of indices to retrieve.
        :return list[dict]: A list of dictionary representations of the rows.
        """
        return [self.retrieve(index) for index in indices]

    def _get_sources(self) -> Sequence["SequenceDataSource[RowT]"]:
        """Internal helper to treat everything as a list of sources."""
        return [self]

    def __add__(self, other: SequenceDataSource[RowT]) -> CombinationSource[RowT]:
        return CombinationSource([*self._get_sources(), *other._get_sources()])


class CombinationSource[RowT: Mapping[str, KnownT]](SequenceDataSource[RowT]):
    """
    Combines multiple data sources into one.
    """

    def __init__(self, sources: Sequence[SequenceDataSource[RowT]]):
        self.sources = sources
        self.cumulative_heights = self._compute_cumulative_heights()

    def _compute_cumulative_heights(self) -> Tensor:
        heights = torch.tensor(
            [0] + [source.height for source in self.sources], dtype=torch.int64
        )
        return torch.cumsum(heights, dim=0)

    @override
    def _get_sources(self) -> Sequence[SequenceDataSource[RowT]]:
        return self.sources

    @property
    @override
    def height(self) -> int:
        return sum(source.height for source in self.sources)

    @override
    def retrieve(self, index: int) -> RowT:
        source_index = int(
            torch.searchsorted(self.cumulative_heights, index, right=True) - 1
        )
        local_index = index - int(self.cumulative_heights[source_index])
        return self.sources[source_index].retrieve(local_index)

    @override
    def batch(self, indices: Sequence[int]) -> list[RowT]:
        # We need to preserve the order of indices
        indices_t = torch.as_tensor(indices, dtype=torch.long)
        # Bucketize indices by source
        source_indices = (
            torch.bucketize(indices_t, self.cumulative_heights, right=True) - 1
        )
        # Compute local indices within each source
        local_indices = indices_t - self.cumulative_heights[source_indices]
        # Prepare output buffer
        results = cast(list[RowT], [None] * len(indices))

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


@final
class TabularSequenceSource[RowT: Mapping[str, KnownT]](SequenceDataSource[RowT]):
    """
    Reads from a tabular data source (e.g., CSV, Excel, Parquet). Stores in a Polars DataFrame.
    """

    def __init__(self, source: Path):
        self.source = source
        self.dataframe = self._read_source()

    def _read_source(self) -> pl.DataFrame:
        match self.source.suffix:
            case ".csv":
                return pl.read_csv(self.source)
            case ".xlsx":
                return pl.read_excel(self.source)
            case ".parquet":
                return pl.read_parquet(self.source)
            case _:
                raise ValueError(f"Unsupported file type: {self.source.suffix}")

    @property
    @override
    def height(self) -> int:
        return self.dataframe.height

    @override
    def retrieve(self, index: int) -> RowT:
        return cast(RowT, self.dataframe.row(index, named=True))

    @override
    def batch(self, indices: Sequence[int]) -> list[RowT]:
        return cast(list[RowT], self.dataframe[indices].to_dicts())


@final
class InMemorySequenceSource[RowT: Mapping[str, KnownT]](SequenceDataSource[RowT]):
    """
    A simple in-memory sequence data source backed by a list of rows.
    """

    def __init__(self, data: pl.DataFrame):
        self.dataframe = data

    @property
    @override
    def height(self) -> int:
        return self.dataframe.height

    @override
    def retrieve(self, index: int) -> RowT:
        return cast(RowT, self.dataframe.row(index, named=True))

    @override
    def batch(self, indices: Sequence[int]) -> list[RowT]:
        return cast(list[RowT], self.dataframe[indices].to_dicts())


@final
class GenomeSliceSource[RowT: Mapping[str, KnownT]](SequenceDataSource[RowT]):
    """
    A data source that retrieves slices of a genome sequence.
    """

    def __init__(
        self,
        genomes_directory: Path,
        metadata_source: Path,
        key_column: str,
        start_column: str | None = None,
        end_column: str | None = None,
        sequence_column: str = "dna_sequence",
    ):
        self.genomes_directory = genomes_directory
        self.metadata = (
            pl.read_parquet(metadata_source)
            if metadata_source.suffix == ".parquet"
            else pl.read_csv(metadata_source)
        )
        self.key_column = key_column
        self.start_column = start_column
        self.end_column = end_column
        self.sequence_column = sequence_column

        if key_column not in self.metadata.columns:
            raise ValueError(f"Key column '{key_column}' not found in metadata.")
        if start_column and start_column not in self.metadata.columns:
            raise ValueError(f"Start column '{start_column}' not found in metadata.")

    @property
    @override
    def height(self) -> int:
        return self.metadata.height

    def _get_genome_path(self, key: str) -> Path:
        path = self.genomes_directory / f"{key}.fasta"
        if not path.is_file():
            raise ValueError(f"Genome file for key '{key}' not found at {path}.")
        return path

    def _load_sequence_uncached(self, key: str) -> str:
        """
        Load a full genome sequence from FASTA.

        :param str key: The key corresponding to the FASTA file.
        :return str: The full genome sequence as a string.
        """
        genome_source = self._get_genome_path(key)

        if not genome_source:
            raise ValueError(f"Genome file for key '{key}' not found.")

        with open(genome_source) as handle:
            rec = next(SeqIO.parse(handle, "fasta"))
            return str(rec.seq)

    @lru_cache(maxsize=32)
    def _load_sequence(self, key: str) -> str:
        return self._load_sequence_uncached(key)

    @override
    def retrieve(self, index: int) -> RowT:
        row = self.metadata.row(index, named=True)
        key = cast(str, row[self.key_column])
        start = int(row[self.start_column]) if self.start_column else None
        end = int(row[self.end_column]) if self.end_column else None

        full_sequence = self._load_sequence(key)

        if start is None or end is None:
            sequence = full_sequence
        else:
            sequence = full_sequence[start : end + 1]

        return cast(RowT, {**row, self.sequence_column: sequence})
