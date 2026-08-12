from typing import Protocol

import torch
from torch import Tensor


class Result(Protocol):
    @property
    def loss(self) -> Tensor: ...


class Objective[ModelT, BatchT, ResultT: Result](Protocol):
    """Pure mathematical transformation task contract."""

    def forward_pass(
        self, model: ModelT, batch: BatchT, device: torch.device
    ) -> ResultT: ...
