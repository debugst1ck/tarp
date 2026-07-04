from typing import TypedDict

from torch import Tensor


class SequenceBatch(TypedDict):
    sequence: Tensor
    attention_mask: Tensor


class ClassificationBatch(SequenceBatch):
    labels: Tensor


class LanguageBatch(SequenceBatch):
    truth: Tensor


class DistillationBatch(TypedDict):
    student: LanguageBatch
    teacher: SequenceBatch


class DiffusionBatch(LanguageBatch):
    timestep_weight: Tensor
