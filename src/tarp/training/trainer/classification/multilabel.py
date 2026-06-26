from collections.abc import Sequence
from typing import override

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import Metric

from tarp.data.datasets.classification.multilabel import MultiLabelClassificationDataset
from tarp.model.tasks.classification import ClassificationModel
from tarp.training.callbacks.core import Callback
from tarp.training.trainer.core import Trainer
from tarp.typed.batch import ClassificationBatch


class MultiLabelClassificationTrainer(
    Trainer[ClassificationModel, ClassificationBatch, Tensor, Tensor]
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
        metrics: Sequence[Metric] = (),
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
            metrics,
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
        self, batch: ClassificationBatch, batch_index: int
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        inputs = batch["sequence"].to(self.context.device)
        labels = batch["labels"].to(self.context.device)
        attention_mask = batch["attention_mask"].to(self.context.device)
        scores, auxillary = self.context.model(inputs, attention_mask=attention_mask)
        loss = self.criterion(scores, labels)
        if auxillary is not None:
            loss = loss + self.criterion(auxillary, labels)
        return loss, scores.detach().cpu(), labels.detach().cpu()

    @override
    def validation_forward(
        self, batch: ClassificationBatch, batch_index: int
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        inputs = batch["sequence"].to(self.context.device)
        labels = batch["labels"].to(self.context.device)
        attention_mask = batch["attention_mask"].to(self.context.device)
        scores, auxillary = self.context.model(inputs, attention_mask=attention_mask)
        loss = self.criterion(scores, labels)
        if auxillary is not None:
            loss = loss + self.criterion(auxillary, labels)
        return loss, scores.detach().cpu(), labels.detach().cpu()
