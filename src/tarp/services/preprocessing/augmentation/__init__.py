from abc import ABC, abstractmethod


class Augmentation(ABC):

    @abstractmethod
    def apply(self, sequence: str) -> str:
        """
        Applies the augmentation technique to the input sequence.

        :param str sequence: The input sequence to augment.
        :return str: The augmented sequence.
        """
        return NotImplementedError


class NoAugmentation(Augmentation):
    def apply(self, sequence: str) -> str:
        """
        Applies no augmentation to the input sequence.

        :param str sequence: The input sequence to augment.
        :return str: The unmodified sequence.
        """
        return sequence
    
    
class CompositeAugmentation(Augmentation):
    def __init__(self, techniques: list[Augmentation]):
        self.techniques = techniques

    def apply(self, sequence: str) -> str:
        for technique in self.techniques:
            sequence = technique.apply(sequence)
        return sequence