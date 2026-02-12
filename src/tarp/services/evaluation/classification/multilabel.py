from collections.abc import Callable, Mapping, Sequence
from typing import Union

import sklearn.metrics
import torch
from torch import Tensor

from tarp.cli.logging import Console


class MultiLabelMetrics:
    """
    Computes multiple metrics for multilabel classification in one call.
    """

    def __init__(self, threshold: Union[float, Tensor] = 0.5, logits: bool = True):
        self.threshold = threshold
        self.logits = logits

        # Registry of metrics (name → function)
        self._metrics: dict[str, Callable[[Tensor, Tensor], float]] = {
            "precision": self._precision,
            "recall": self._recall,
            "f1": self._f1,
            "subset_accuracy": self._subset_accuracy,
            "roc_auc": self._roc_auc,
            "hamming_loss": self._hamming_loss,
            "label_ranking_average_precision": self._label_ranking_average_precision,
            "mathews_correlation_coefficient": self._mathews_correlation_coefficient,
            "mean_probability": self._mean_probability,
        }

    def _predict_probability(self, logits: Tensor) -> Tensor:
        """Convert logits to probabilities via sigmoid."""
        return torch.sigmoid(logits).detach()

    def _predict(self, logits: Tensor) -> Tensor:
        """Binarize probabilities at threshold."""
        if not self.logits:
            return logits
        probabilities = self._predict_probability(logits)

        if isinstance(self.threshold, float) or isinstance(self.threshold, int):
            # Broadcastable threshold
            threshold_tensor = torch.full_like(probabilities, self.threshold)
        else:
            threshold_tensor = self.threshold

        return (probabilities > threshold_tensor).int()

    # --- individual metric implementations ---
    def _precision(self, logits: Tensor, targets: Tensor) -> float:
        predictions = self._predict(logits)
        return sklearn.metrics.precision_score(
            targets.cpu().numpy(),
            predictions.cpu().numpy(),
            average="micro",
            zero_division=0,  # type: ignore
        )

    def _recall(self, logits: Tensor, targets: Tensor) -> float:
        predictions = self._predict(logits)
        return sklearn.metrics.recall_score(
            targets.cpu().numpy(),
            predictions.cpu().numpy(),
            average="micro",
            zero_division=0,  # type: ignore
        )

    def _f1(self, logits: Tensor, targets: Tensor) -> float:
        predictions = self._predict(logits)
        return sklearn.metrics.f1_score(
            targets.cpu().numpy(),
            predictions.cpu().numpy(),
            average="micro",
            zero_division=0,  # type: ignore
        )

    def _subset_accuracy(self, logits: Tensor, targets: Tensor) -> float:
        predictions = self._predict(logits)
        return sklearn.metrics.accuracy_score(
            targets.cpu().numpy(), predictions.cpu().numpy()
        )

    def _roc_auc(self, logits: Tensor, targets: Tensor) -> float:
        if not self.logits:
            Console.warning("ROC AUC metric expects logits, but got probabilities.")
            return float("nan")

        probs = self._predict_probability(logits).cpu().numpy()
        y_true = targets.cpu().numpy()

        # Find valid classes (those with at least one positive sample)
        valid_classes = [
            i for i in range(y_true.shape[1]) if len(set(y_true[:, i])) > 1
        ]

        # If no valid classes, return NaN
        if not valid_classes:
            Console.warning("No valid classes for ROC AUC computation.")
            return float("nan")

        if len(valid_classes) < y_true.shape[1]:
            skipped = y_true.shape[1] - len(valid_classes)
            Console.warning(
                f"Skipping {skipped} invalid classes (all-zeros or all-ones) for ROC AUC."
            )

        return float(
            sklearn.metrics.roc_auc_score(
                y_true[:, valid_classes], probs[:, valid_classes], average="macro"
            )
        )

    def _mathews_correlation_coefficient(
        self, logits: Tensor, targets: Tensor
    ) -> float:
        """
        Matthews Correlation Coefficient for multilabel classification.
        Uses macro-average across labels.
        """
        predictions = self._predict(logits)

        return sklearn.metrics.matthews_corrcoef(
            targets.cpu().numpy().ravel(),
            predictions.cpu().numpy().ravel(),
        )

    def _hamming_loss(self, logits: Tensor, targets: Tensor) -> float:
        predictions = self._predict(logits)
        return sklearn.metrics.hamming_loss(
            targets.cpu().numpy(), predictions.cpu().numpy()
        )

    def _label_ranking_average_precision(
        self, logits: Tensor, targets: Tensor
    ) -> float:
        if not self.logits:
            Console.warning(
                "Label Ranking Average Precision metric expects logits, but got probabilities."
            )
            return float("nan")

        probs = self._predict_probability(logits).cpu().numpy()
        y_true = targets.cpu().numpy()

        return sklearn.metrics.label_ranking_average_precision_score(y_true, probs)

    def _mean_probability(self, logits: Tensor, targets: Tensor) -> float:
        """
        Mean predicted probability conditioned on positive labels.
        """
        if not self.logits:
            Console.warning(
                "Mean Probability metric expects logits, but got probabilities."
            )
            return float("nan")

        probs = self._predict_probability(logits)

        # Sum probabilities over true labels per sample
        sum_true_label_probs = (probs * targets).sum(dim=1)

        # Count true labels per sample
        num_true_labels = targets.sum(dim=1)

        # Keep only samples with at least one positive label
        mask = num_true_labels > 0
        if not mask.any():
            return float("nan")

        mean_probs_per_sample = sum_true_label_probs[mask] / num_true_labels[mask]

        return mean_probs_per_sample.mean().item()

    def compute(
        self,
        logits: Union[Tensor, Sequence[Tensor]],
        targets: Union[Tensor, Sequence[Tensor]],
    ) -> Mapping[str, float]:
        """
        Compute all metrics at once and return as dict.
        """
        # torch.cat can only concatenate tensors if list or tuple
        # Tensors are used as is
        # Other sequences are cast to List[Tensor] and concatenated
        if isinstance(logits, Tensor):
            logits = logits
        else:
            logits = torch.cat(list(logits), dim=0)
        if isinstance(targets, Tensor):
            targets = targets
        else:
            targets = torch.cat(list(targets), dim=0)
        return {name: fn(logits, targets) for name, fn in self._metrics.items()}

    def add_metric(self, name: str, fn: Callable[[Tensor, Tensor], float]) -> None:
        """
        Allow user to register a custom metric.
        """
        self._metrics[name] = fn

    def remove_metric(self, name: str) -> None:
        """
        Allow user to remove a registered metric.
        """
        self._metrics.pop(name, None)
