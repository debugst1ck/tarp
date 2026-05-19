from typing import Literal

from torch import Tensor, nn


class Criterion(nn.Module):
    def __init__(self, reduction: Literal["mean", "sum", "none"] = "mean"):
        super().__init__()
        self.reduction: Literal["mean", "sum", "none"] = reduction

    def _apply_reduction(self, loss: Tensor) -> Tensor:
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
