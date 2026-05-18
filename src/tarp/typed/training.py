from typing import TypeVar

from torch import Tensor, nn

ModelT = TypeVar("ModelT", bound=nn.Module)
PredictionT = TypeVar("PredictionT", bound=Tensor)
TargetT = TypeVar("TargetT", bound=Tensor)
