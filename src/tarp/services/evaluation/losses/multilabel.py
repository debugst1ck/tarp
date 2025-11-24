from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tarp.services.evaluation import Reduction


class AsymmetricFocalLoss(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 1,
        clip: float = 0.05,
        epsilon: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True,
        reduction: Reduction = Reduction.MEAN,
        class_weights: Optional[Tensor] = None,
    ):
        """
        Asymmetric Loss for multi-label classification.

        :param float gamma_neg: focusing parameter for negative samples, higher values put more focus on hard negatives
        :param float gamma_pos: focusing parameter for positive samples, higher values put more focus on hard positives
        :param float clip: if > 0, adds a small value to the negative logits before applying sigmoid, helps with extreme negatives
        :param float epsilon: small value to avoid log(0)
        :param bool disable_torch_grad_focal_loss: if True, disables gradient computation for focal loss part to save memory
        :param str reduction: reduction method to apply to the output loss ('none', 'mean', 'sum')
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.epsilon = epsilon
        self.reduction = reduction
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Asymmetric Loss for multi-label classification.
        Args:
            logits: raw model outputs (before sigmoid), shape (batch, num_labels)
            targets: binary targets, same shape
        """
        # Use 'probs' for the sigmoid output to indicate they are probabilities
        probs = torch.sigmoid(logits)

        # Clearly distinguish positive and negative probabilities
        probs_pos = probs
        probs_neg: Tensor = 1 - probs

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            # Use 'clamped_probs_neg' to show the effect of clipping
            probs_neg = (probs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        # Use 'log_likelihood' to describe the loss components
        log_likelihood_pos = targets * torch.log(probs_pos.clamp(min=self.epsilon))
        log_likelihood_neg: Tensor = (1 - targets) * torch.log(
            probs_neg.clamp(min=self.epsilon)
        )

        # Combine the positive and negative parts into a single loss
        loss = log_likelihood_pos + log_likelihood_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)

            # Use 'pt' for probability-target products, a common notation
            pt_pos = probs_pos * targets
            pt_neg = probs_neg * (1 - targets)
            pt = pt_pos + pt_neg

            # 'gamma_for_each_class' is more descriptive than 'one_sided_gamma'
            gamma_for_each_class = self.gamma_pos * targets + self.gamma_neg * (
                1 - targets
            )

            # 'focusing_factor' is a better name for the weight
            focusing_factor = torch.pow(1 - pt, gamma_for_each_class)

            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)

            loss *= focusing_factor

        # Apply class weights
        if self.class_weights is not None:
            loss *= self.class_weights.to(loss.device).unsqueeze(0)

        # Apply reduction
        loss = -loss
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
