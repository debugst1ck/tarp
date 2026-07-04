from typing import Literal

from torch import Tensor, nn


class Criterion(nn.Module):
    def __init__(self, reduction: Literal["mean", "sum", "none"] = "mean"):
        super().__init__()
        self.reduction: Literal["mean", "sum", "none"] = reduction

    def _apply_reduction(self, loss: Tensor) -> Tensor:
        match self.reduction:
            case "none":
                return loss
            case "mean":
                return loss.mean()
            case "sum":
                return loss.sum()
