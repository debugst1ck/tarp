from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tarp.services.evaluation import Reduction


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss for Multi-Label Classification.

    Reference:
    - Improving Object Detection with One-Sided Unsupervised Domain Adaptation [Bodla et al., 2017](https://arxiv.org/abs/1708.02002)
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        reduction: Reduction = Reduction.MEAN,
        class_weights: Optional[Tensor] = None,
    ):
        """
        :param float gamma_neg: Focusing parameter for negative samples.
        :param float gamma_pos: Focusing parameter for positive samples.
        :param float clip: Optional clipping value for logits of negative samples.
        :param Reduction reduction: Reduction method to apply to the output.
        :param class_weights: Optional tensor of shape [num_classes] for per-class weighting.
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction

        self.class_weights: Optional[Tensor]  # Typing for pyright
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        # Optional asymmetric clipping on logits
        if self.clip > 0:
            logits = torch.where(
                targets == 0,
                logits + self.clip,
                logits,
            )
        # Stable BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Apply class weights if provided -> shape [batch_size, num_classes]
        if self.class_weights is not None:
            bce = bce * self.class_weights.unsqueeze(0)

        probabilities = torch.sigmoid(logits)

        # Probabilities of the true class
        probabilities_true = probabilities * targets + (1 - probabilities) * (
            1 - targets
        )
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)

        # Focal weights should NOT receive gradients
        with torch.no_grad():
            focal_weight = torch.pow(1 - probabilities_true, gamma)

        loss = focal_weight * bce
        match self.reduction:
            case Reduction.MEAN:
                return loss.mean()
            case Reduction.SUM:
                return loss.sum()
            case _:
                return loss


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, torch.Tensor]] = None,
        reduction: Reduction = Reduction.MEAN,
        logits: bool = True,
    ):
        """
        Focal Loss for binary or multi-label classification.

        :param gamma: focusing parameter that reduces the loss contribution from easy examples
        :param alpha: balancing factor.
                      - If float in [0,1], scalar positive/negative weighting.
                      - If Tensor of shape [num_classes], per-class weights.
        :param reduction: 'none' | 'mean' | 'sum'
        :param logits: if True, expects raw logits; otherwise expects probabilities.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.logits = logits

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            probs = torch.sigmoid(logits)
        else:
            bce_loss = F.binary_cross_entropy(logits, targets, reduction="none")
            probs = logits

        # pt is probability of the true class
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_factor = (1 - pt) ** self.gamma
        focal_loss = focal_factor * bce_loss

        # Apply alpha (scalar or per-class tensor)
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # Make sure alpha is on the same device
                alpha_factor = self.alpha.to(logits.device).unsqueeze(
                    0
                )  # [1, num_classes]
                alpha_factor = alpha_factor * targets + (1 - alpha_factor) * (
                    1 - targets
                )
            else:
                # Scalar case
                alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)

            focal_loss = alpha_factor * focal_loss

        # Reduction
        match self.reduction:
            case Reduction.MEAN:
                return focal_loss.mean()
            case Reduction.SUM:
                return focal_loss.sum()
            case _:
                return focal_loss
