# Sequence augmentation techniques
import random
from Bio.Seq import Seq
from array import array
import sys
from typing import Optional

from tarp.services.preprocessing.augmentation import Augmentation


# Stochastic Augmentation Techniques
class ReverseComplement(Augmentation):
    def __init__(self, complement_rate: float = 1.0):
        self.complement_rate = complement_rate

    def apply(self, sequence: str) -> str:
        """
        Applies the reverse complement augmentation to the input sequence.
        This simulates the opposite strand in DNA sequencing.

        :param str sequence: The input sequence to augment.
        :return str: The augmented sequence.
        """
        if random.random() > self.complement_rate:
            return sequence
        bio_seq = Seq(sequence)
        return str(bio_seq.reverse_complement())


class RandomMutation(Augmentation):
    def __init__(
        self, mutation_rate: float = 0.01, vocabulary: list[str] = ["A", "C", "G", "T"]
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


class InsertionDeletion(Augmentation):
    def __init__(
        self,
        insertion_rate: float = 0.01,
        deletion_rate: float = 0.01,
        max_insert_size: int = 3,
        max_delete_size: int = 3,
        gc_ratio: Optional[float] = None,
    ):
        self.insertion_rate = insertion_rate
        self.deletion_rate = deletion_rate
        self.max_insert_size = max_insert_size
        self.max_delete_size = max_delete_size
        self.gc_ratio = gc_ratio

        self.at_bases = ["A", "T"]
        self.gc_bases = ["G", "C"]

    def _sample_insertion(self, length: int, gc_ratio: float) -> list[str]:
        """
        Sample insertion bases with given GC ratio.
        """
        return [
            (
                random.choice(self.gc_bases)
                if random.random() < gc_ratio
                else random.choice(self.at_bases)
            )
            for _ in range(length)
        ]

    def apply(self, sequence: str) -> str:
        """
        Applies the indel augmentation to the input sequence.
        Insertions respect GC ratio; deletions remove contiguous segments.

        :param str sequence: The input sequence to augment.
        :return str: The augmented sequence.
        """
        # Decide gc_ratio dynamically if not provided
        gc_ratio = self.gc_ratio
        if gc_ratio is None:
            gc_count = sum(1 for base in sequence if base in self.gc_bases)
            at_count = sum(1 for base in sequence if base in self.at_bases)
            gc_ratio = gc_count / max(gc_count + at_count, 1)

        augmented = []
        i = 0
        while i < len(sequence):
            # Deletion
            if random.random() < self.deletion_rate:
                delete_length = random.randint(1, self.max_delete_size)
                i += delete_length
                continue  # skip adding deleted bases

            # Keep original base
            augmented.append(sequence[i])

            # Insertion
            if random.random() < self.insertion_rate:
                insert_length = random.randint(1, self.max_insert_size)
                augmented.extend(self._sample_insertion(insert_length, gc_ratio))

            i += 1

        return "".join(augmented)
