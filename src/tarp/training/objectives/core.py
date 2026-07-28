from typing import Protocol

import torch
from torch import Tensor


class Result(Protocol):
    loss: Tensor


class Objective[ModelT, BatchT, ResultT: Result](Protocol):
    """Pure mathematical transformation task contract."""

    def forward_pass(
        self, model: ModelT, batch: BatchT, device: torch.device
    ) -> ResultT: ...
