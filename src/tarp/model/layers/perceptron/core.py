from abc import ABC, abstractmethod
from typing import override

from torch import Tensor, nn


class FeedForward(nn.Module, ABC):
    @override
    @abstractmethod
    def forward(self, features: Tensor) -> Tensor: ...
