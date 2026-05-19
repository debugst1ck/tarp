from typing import Literal, final, override

import torch
import torch.nn.functional as F
from torch import Tensor

from tarp.model.criterion.core import Criterion


@final
class LabelDistributionAwareMarginLoss(Criterion):
    def __init__(
        self,
        class_counts: Tensor,
        maximum_margin: float = 0.5,
        tau: float = 1.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__(reduction)
        self.maximum_margin = maximum_margin
        self.tau = tau
        counts = class_counts.float()
        counts = torch.where(counts > 0, counts, torch.ones_like(counts))
        margins = 1.0 / (torch.sqrt(torch.sqrt(counts)) + torch.finfo(counts.dtype).eps)
        if margins.max() > 0:
            margins = margins * (maximum_margin / margins.max())
        self.margins: Tensor
        self.register_buffer("margins", self.margins)

    @override
    def forward(self, scores: Tensor, targets: Tensor) -> Tensor:
        targets = targets.to(scores.dtype)

        adjusted_scores = scores - self.margins.unsqueeze(0) * targets
        scaled_scores = adjusted_scores * self.tau

        loss = F.binary_cross_entropy_with_logits(
            scaled_scores,
            targets,
            reduction="none",
        )
        return self._apply_reduction(loss)
