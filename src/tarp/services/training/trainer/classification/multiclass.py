from typing import Optional

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset

from tarp.model.finetuning.classification import ClassificationModel
from tarp.services.evaluation.classification.multilabel import MultiLabelMetrics
from tarp.services.training.trainer import Trainer


class MultiClassClassificationTrainer(Trainer):
    def __init__(
        self,
        model: ClassificationModel,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        device: torch.device,
        batch_size: int = 32,
        epochs: int = 10,
        max_grad_norm: float = 1.0,
        num_workers: int = 0,
        use_amp: bool = True,
        class_weights: Optional[Tensor] = None,
        criterion: Optional[nn.Module] = None,
        accumulation_steps: int = 1,
    ):
        if criterion is None:
            if class_weights is not None:
                self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            batch_size=batch_size,
            epochs=epochs,
            max_grad_norm=max_grad_norm,
            num_workers=num_workers,
            use_amp=use_amp,
            accumulation_steps=accumulation_steps,
        )

        self.criterion = self.criterion.to(device)

        self.metrics = (
            MultiLabelMetrics()
        )  # Reuse MultiLabelMetrics for multi-class as well

    def training_step(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        inputs = batch["sequence"].to(self.context.device)
        labels = batch["labels"].to(self.context.device)
        attention_mask = batch["attention_mask"].to(self.context.device)
        logits: Tensor = self.context.model(inputs, attention_mask=attention_mask)
        loss = self.criterion(logits, labels)
        return loss, logits.detach().cpu(), labels.detach().cpu()

    def validation_step(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        inputs = batch["sequence"].to(self.context.device)
        labels = batch["labels"].to(self.context.device)
        attention_mask = batch["attention_mask"].to(self.context.device)
        logits: Tensor = self.context.model(inputs, attention_mask=attention_mask)
        loss = self.criterion(logits, labels)
        return loss, logits.detach().cpu(), labels.detach().cpu()

    def compute_metrics(
        self, prediction: list[Tensor], expected: list[Tensor]
    ) -> dict[str, float]:
        return self.metrics.compute(prediction, expected)
