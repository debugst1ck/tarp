import random
from typing import Optional
import sys
from array import array

from tarp.services.preprocessing.augmentation import Augmentation


class InsertionDeletion(Augmentation):
    def __init__(
        self,
        insertion_rate: float = 0.01,
        deletion_rate: float = 0.01,
        max_insert_size: int = 3,
        max_delete_size: int = 3,
        hydrophobic_ratio: Optional[float] = None,
    ):
        self.insertion_rate = insertion_rate
        self.deletion_rate = deletion_rate
        self.max_insert_size = max_insert_size
        self.max_delete_size = max_delete_size
        self.hydrophobic_ratio = hydrophobic_ratio

        # Amino acid classes
        self.hydrophobic = list("AVILMFYW")
        self.hydrophilic = list("STNQKRHDEC")  # polar + charged

    def _sample_insertion(self, length: int, ratio: float) -> list[str]:
        """Sample insertion residues based on hydrophobic ratio."""
        return [
            (
                random.choice(self.hydrophobic)
                if random.random() < ratio
                else random.choice(self.hydrophilic)
            )
            for _ in range(length)
        ]

    def apply(self, sequence: str) -> str:
        """Insertions/deletions respecting hydrophobic composition."""
        ratio = self.hydrophobic_ratio
        if ratio is None:
            hydrophobic_count = sum(1 for aa in sequence if aa in self.hydrophobic)
            hydrophilic_count = sum(1 for aa in sequence if aa in self.hydrophilic)
            ratio = hydrophobic_count / max(hydrophobic_count + hydrophilic_count, 1)

        augmented = []
        i = 0
        while i < len(sequence):
            # Deletion
            if random.random() < self.deletion_rate:
                delete_length = random.randint(1, self.max_delete_size)
                i += delete_length
                continue

            augmented.append(sequence[i])

            # Insertion
            if random.random() < self.insertion_rate:
                insert_length = random.randint(1, self.max_insert_size)
                augmented.extend(self._sample_insertion(insert_length, ratio))

            i += 1

        return "".join(augmented)


class RandomMutation(Augmentation):
    def __init__(
        self,
        mutation_rate: float = 0.01,
        vocabulary: list[str] = list("ACDEFGHIKLMNPQRSTVWY"),
    ):
        self.mutation_rate = mutation_rate
        self.vocabulary = vocabulary

    def apply(self, sequence: str) -> str:
        """
        Mutates a sequence by randomly changing characters based on the mutation rate.

        :param str sequence: The input sequence to mutate.
        :return str: The mutated sequence.
        """
        # Make a mutable array
        # If python version is less than 3.13
        if sys.version_info < (3, 13):
            # Deprecate "u" wchar_t in 3.13 onwards
            array_seq = array("u", sequence)
        else:
            # Replaced by "w" Py_UCS4 from 3.16
            array_seq = array("w", sequence)

        for i in range(len(array_seq)):
            if random.random() < self.mutation_rate:
                array_seq[i] = random.choice(self.vocabulary)
        return array_seq.tounicode()