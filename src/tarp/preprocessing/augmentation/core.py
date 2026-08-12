from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import final, override


class Augmentation(ABC):
    @abstractmethod
    def apply(self, sequence: str) -> str:
        """
        Applies the augmentation technique to the input sequence.

        :param str sequence: The input sequence to augment.
        :return str: The augmented sequence.
        """
        raise NotImplementedError


@final
class NoAugmentation(Augmentation):
    @override
    def apply(self, sequence: str) -> str:
        """
        Returns the input sequence unchanged.

        :param str sequence: The input sequence to augment.
        :return str: The unchanged input sequence.
        """
        return sequence


@final
class CompositeAugmentation(Augmentation):
    def __init__(self, techniques: Iterable[Augmentation]):
        self.techniques = techniques

    @override
    def apply(self, sequence: str) -> str:
        for technique in self.techniques:
            sequence = technique.apply(sequence)
        return sequence
