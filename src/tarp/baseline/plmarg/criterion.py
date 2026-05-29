from collections.abc import Sequence
from typing import Literal, final, override

import torch
from torch import Tensor

from tarp.model.criterion.core import Criterion


@final
class AntiMicrobialResistanceGeneCriterion(Criterion):
    def __init__(
        self,
        criterion: Criterion,
        minority_class_indices: Sequence[int],
        non_amr_index: int,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__(reduction)
        if criterion.reduction != "none":
            raise ValueError(
                "The provided criterion must have reduction set to 'none'."
            )

        super().__init__(reduction)

        self.minority_class_indices: Tensor
        self.register_buffer(
            "minority_indices", torch.tensor(minority_class_indices, dtype=torch.long)
        )
        self.non_amr_index = non_amr_index
        self.criterion = criterion

    @override
    def forward(self, scores: Tensor, targets: Tensor) -> Tensor:
        losses = self.criterion(scores, targets)  # [B, C]

        non_amr_loss = losses[:, self.non_amr_index]  # [B]
        minority_losses = losses[:, self.minority_indices]  # [B, M]

        # Create the Dynamic AMR Mask
        # If non_amr == 1, then AMR_mask == 0 (Do not calculate drug classes)
        # If non_amr == 0, then AMR_mask == 1 (Calculate specific drug resistance)
        amr_sample_mask = (1.0 - targets[:, self.non_amr_index]).unsqueeze(1)  # [B, 1]

        masked_minority_loss = minority_losses * amr_sample_mask

        match self.reduction:
            case "none":
                return torch.cat(
                    [non_amr_loss.unsqueeze(1), masked_minority_loss], dim=1
                )
            case "mean":
                non_amr_loss_reduced = non_amr_loss.mean()
                total_active_minority_elements = (
                    amr_sample_mask.sum() * minority_losses.size(1)
                )
                minority_loss_reduced = masked_minority_loss.sum() / (
                    total_active_minority_elements
                    + torch.finfo(masked_minority_loss.dtype).eps
                )
                return non_amr_loss_reduced + minority_loss_reduced
            case "sum":
                return non_amr_loss.sum() + masked_minority_loss.sum()
