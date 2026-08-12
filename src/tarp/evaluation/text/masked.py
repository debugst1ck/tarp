# This is a wrapper for text generation evaluation metrics that require sequence-level inputs, such as MLM Accuracy. The `metrics.Accuracy` class requires the shape of (batch_size, num_classes) and not (batch_size, sequence_length, num_classes), so this wrapper reshapes the inputs accordingly.

from typing import override

from torch import Tensor
from torchmetrics.classification.accuracy import MulticlassAccuracy


class MaskedLanguageAccuracy(MulticlassAccuracy):
    @override
    def update(self, preds: Tensor, target: Tensor) -> None:
        # Reshape the inputs to (batch_size * sequence_length, num_classes) and (batch_size * sequence_length)
        preds = preds.reshape(-1, preds.shape[-1])
        target = target.reshape(-1)
        super().update(preds, target)
