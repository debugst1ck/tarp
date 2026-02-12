# Generic typing for trainer components.
#
from typing import TypeVar

PredT = TypeVar("PredT")  # prediction type
TargetT = TypeVar("TargetT")  # target/label type
BatchT = TypeVar(
    "BatchT", contravariant=True
)  # batch type (usually a tuple of tensors)
