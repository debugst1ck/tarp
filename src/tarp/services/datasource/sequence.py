import polars as pl
from pathlib import Path
from typing import Optional
from Bio import SeqIO
from abc import ABC, abstractmethod
from functools import lru_cache
from Bio.Seq import Seq
import torch
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

    def batch(self, indices: list[int]) -> list[dict]:
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

    def batch(self, indices: list[int]) -> list[dict]:
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
            [source.height for source in self.sources], dtype=torch.long
        )
        return torch.cumsum(heights, dim=0)

    @property
    def height(self) -> int:
        return sum(source.height for source in self.sources)

    def retrieve(self, index: int) -> dict:
        source_index, local_index = self._get_source_and_local_index(index)
        if 0 <= source_index < len(self.sources):
            return self.sources[source_index].retrieve(local_index)
        raise IndexError("Index out of range for combined data sources.")

    def _get_source_and_local_index(self, index: int) -> tuple[int, int]:
        # bounds check
        if index < 0 or index >= int(self.height):
            raise IndexError(f"Index {index} out of range (0..{self.height-1})")

        # make a scalar tensor on same device as cumulative heights
        index: Tensor = torch.tensor(
            index,
            device=self._cumulative_heights.device,
            dtype=self._cumulative_heights.dtype,
        )
        source_index = torch.bucketize(index, self._cumulative_heights, right=True)
        
        previous_cumulative_height = torch.where(
            source_index == 0,
            torch.tensor(0, device=self._cumulative_heights.device),
            self._cumulative_heights[source_index - 1],
        )
        
        local_index = index - previous_cumulative_height

        return int(source_index.item()), int(local_index.item())

    def batch(self, indices: list[int]) -> list[dict]:
        """
        Efficiently batch indices across multiple sources, preserving input order.
        """
        if not indices:
            return []

        indices: Tensor = torch.as_tensor(indices)

        source_indices: Tensor = torch.bucketize(
            indices, self._cumulative_heights, right=True
        )
        previous_cumulative_height = torch.cat([torch.tensor([0]), self._cumulative_heights])
        local_indices = indices - previous_cumulative_height[source_indices]

        results = [None] * len(indices)

        unique_sources: Tensor = torch.unique(source_indices)
        for source_index in unique_sources:
            mask = source_indices == source_index
            positions = torch.nonzero(mask, as_tuple=False).squeeze(1)
            source_local_indices = local_indices[mask].tolist()
            batch_results = self.sources[source_index.item()].batch(
                source_local_indices
            )

            for position, result in zip(positions.tolist(), batch_results):
                results[position] = result

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

    def batch(self, indices: list[int]) -> list[dict]:
        return self.dataframe.rows(indices, named=True)


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
        orientation_column: Optional[str] = None,
        sequence_column: str = "sequence",
    ):
        self.directory = directory
        self.metadata = metadata
        self.key_column = key_column
        self.start_column = start_column
        self.end_column = end_column
        self.orientation_column = orientation_column
        self.sequence_column = sequence_column

        self.df = (
            pl.read_parquet(metadata)
            if metadata.suffix == ".parquet"
            else pl.read_csv(metadata)
        )

        if self.key_column not in self.df.columns:
            raise ValueError(f"Key column {self.key_column} not found in {self.df.columns}.")

        if self.start_column not in self.df.columns:
            Console.warning(f"Start column {self.start_column} not found in metadata.")

        if self.end_column not in self.df.columns:
            Console.warning(f"End column {self.end_column} not found in metadata.")

        self._fasta_map = {p.stem: p for p in self.directory.glob("*.fasta")}

    @lru_cache(maxsize=32)
    def _load_sequence(self, key: str) -> Seq:
        return self._load_sequence_uncached(key)

    def _load_sequence_uncached(self, key: str) -> Seq:
        """
        Load a full genome sequence from FASTA.

        :param str key: The key corresponding to the FASTA file.
        :return str: The full genome sequence as a string.
        """
        fasta_path = self._fasta_map.get(key)
        if not fasta_path:
            raise FileNotFoundError(f"No FASTA found for {key}")

        with open(fasta_path) as handle:
            return [rec.seq for rec in SeqIO.parse(handle, "fasta")][0]

    @property
    def height(self) -> int:
        return self.df.height

    def retrieve(self, index: int) -> dict:
        row = self.df.row(index, named=True)
        key = row[self.key_column]
        start = row[self.start_column] if self.start_column in row else None
        end = row[self.end_column] if self.end_column in row else None
        orientation = row.get(self.orientation_column, "+")

        if key not in self._fasta_map:
            raise FileNotFoundError(f"No FASTA found for {key}")

        full_sequence = self._load_sequence(key)

        if start is None and end is None:
            sequence = full_sequence
        else:
            sequence = full_sequence[start:end]

        if orientation == "-":
            sequence = sequence.reverse_complement()

        row[self.sequence_column] = str(sequence)
        return row

    def batch(self, indices: list[int]) -> list[dict]:
        subset = self.df[indices]
        groups = subset.partition_by(self.key_column, as_dict=True)

        results = []
        for key, group in groups.items():
            if key[0] not in self._fasta_map:
                raise FileNotFoundError(f"No FASTA found for {key[0]}")

            full_sequence = self._load_sequence(key[0])

            for row in group.rows(named=True):
                start = row[self.start_column] if self.start_column in row else None
                end = row[self.end_column] if self.end_column in row else None
                orientation = row.get(self.orientation_column, "+")

                if start is None and end is None:
                    sequence = full_sequence
                else:
                    sequence = full_sequence[start:end]

                if orientation == "-":
                    sequence = sequence.reverse_complement()

                row[self.sequence_column] = str(sequence)
                results.append(row)
        if not len(results) == len(indices):
            Console.warning(
                f"Batch retrieval returned {len(results)} results, "
                f"but {len(indices)} were requested."
            )
            exit(1)
        return results
