from typing import Protocol

import torch
from torch import Tensor


class Objective[ModelT, BatchT, PredictionT, TargetT](Protocol):
    """Pure mathematical transformation task contract."""

    def forward_pass(
        self, model: ModelT, batch: BatchT, device: torch.device
    ) -> tuple[Tensor, PredictionT, TargetT]: ...
