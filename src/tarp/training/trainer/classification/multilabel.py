from collections.abc import Sequence
from typing import override

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from tarp.data.datasets.classification.multilabel import MultiLabelClassificationDataset
from tarp.functional.evaluation.classification import (
    accuracy,
    macro_f1_score,
    precision,
    recall,
    top_k_accuracy,
)
from tarp.model.heads.classification import ClassificationModel
from tarp.training.callbacks.core import Callback
from tarp.training.trainer.core import Trainer


class MultiLabelClassificationTrainer(
    Trainer[ClassificationModel, dict[str, Tensor], Tensor, Tensor]
):
    def __init__(
        self,
        model: ClassificationModel,
        training_dataset: MultiLabelClassificationDataset,
        validation_dataset: MultiLabelClassificationDataset,
        optimizer: Optimizer,
        device: torch.device,
        criterion: nn.Module,
        scheduler: LRScheduler | None = None,
        batch_size: int = 32,
        epochs: int = 10,
        gradient_clipping_threshold: float = 1.0,
        worker_count: int = 0,
        mixed_precision: bool = True,
        accumulation_steps: int = 1,
        persistent_workers: bool = True,
        callbacks: Sequence[Callback] | None = None,
        shared: dict[str, object] | None = None,
    ):
        super().__init__(
            model,
            training_dataset,
            validation_dataset,
            optimizer,
            device,
            scheduler,
            batch_size,
            epochs,
            gradient_clipping_threshold,
            worker_count,
            mixed_precision,
            accumulation_steps,
            persistent_workers,
            callbacks,
            shared,
        )
        self.criterion = criterion

    @override
    def training_forward(
        self, batch: dict[str, Tensor], batch_index: int
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        inputs = batch["sequence"].to(self.context.device)
        labels = batch["labels"].to(self.context.device)
        attention_mask = batch["attention_mask"].to(self.context.device)
        scores, auxillary = self.context.model.forward(
            inputs, attention_mask=attention_mask, payload_mask=attention_mask
        )
        loss = self.criterion(scores, labels)
        if auxillary is not None:
            loss = loss + self.criterion(auxillary, labels)
        return loss, scores.detach().cpu(), labels.detach().cpu()

    @override
    def validation_forward(
        self, batch: dict[str, Tensor], batch_index: int
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        inputs = batch["sequence"].to(self.context.device)
        labels = batch["labels"].to(self.context.device)
        attention_mask = batch["attention_mask"].to(self.context.device)
        scores, auxillary = self.context.model.forward(
            inputs, attention_mask=attention_mask, payload_mask=attention_mask
        )
        loss = self.criterion(scores, labels)
        if auxillary is not None:
            loss = loss + self.criterion(auxillary, labels)
        return loss, scores.detach().cpu(), labels.detach().cpu()

    @override
    def compute_metrics(
        self, predictions: Sequence[Tensor], targets: Sequence[Tensor], top_k: int = 2
    ) -> dict[str, float]:
        predictions_t = torch.cat(list(predictions), dim=0)
        targets_t = torch.cat(list(targets), dim=0)

        # Threshold based metrics say 0.5
        threshold = 0.5
        binary_predictions = (predictions_t >= threshold).float()
        binary_targets = targets_t.float()

        return {
            "accuracy": accuracy(binary_predictions, binary_targets).item(),
            "precision": precision(binary_predictions, binary_targets).item(),
            "recall": recall(binary_predictions, binary_targets).item(),
            "f1_score": macro_f1_score(binary_predictions, binary_targets).item(),
            "top_k_accuracy": top_k_accuracy(predictions_t, targets_t, k=top_k).item(),
        }
