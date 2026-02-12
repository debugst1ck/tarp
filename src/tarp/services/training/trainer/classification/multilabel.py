from collections.abc import Mapping, Sequence
from typing import Optional

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from tarp.model.finetuning.classification import ClassificationModel
from tarp.services.datasets.classification.multilabel import (
    MultiLabelClassificationDataset,
)
from tarp.services.evaluation.classification.multilabel import MultiLabelMetrics
from tarp.services.evaluation.losses.multilabel import AsymmetricFocalLoss
from tarp.services.training.trainer import Trainer


class MultiLabelClassificationTrainer(Trainer[dict[str, Tensor], Tensor, Tensor]):
    def __init__(
        self,
        model: ClassificationModel,
        train_dataset: MultiLabelClassificationDataset,
        valid_dataset: MultiLabelClassificationDataset,
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
        persistent_workers: bool = False,
    ):
        if criterion is None:
            if class_weights is not None:
                # self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
                self.criterion = AsymmetricFocalLoss(
                    gamma_neg=2, gamma_pos=0, class_weights=class_weights.to(device)
                )
            else:
                self.criterion = nn.BCEWithLogitsLoss()
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
            persistent_workers=persistent_workers,
        )

        self.criterion = self.criterion.to(device)
        self.labels = train_dataset.label_columns

    def training_forward(
        self, batch: dict[str, Tensor], batch_index: int
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        inputs = batch["sequence"].to(self.context.device)
        labels = batch["labels"].to(self.context.device)
        attention_mask = batch["attention_mask"].to(self.context.device)
        logits: Tensor = self.context.model(inputs, attention_mask=attention_mask)
        loss = self.criterion(logits, labels)
        return loss, logits.detach().cpu(), labels.detach().cpu()

    def validation_step(
        self, batch: dict[str, Tensor], batch_index: int
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        inputs = batch["sequence"].to(self.context.device)
        labels = batch["labels"].to(self.context.device)
        attention_mask = batch["attention_mask"].to(self.context.device)
        logits: Tensor = self.context.model(inputs, attention_mask=attention_mask)
        loss = self.criterion(logits, labels)
        return loss, logits.detach().cpu(), labels.detach().cpu()

    def compute_metrics(
        self, prediction: Sequence[Tensor], expected: Sequence[Tensor]
    ) -> Mapping[str, float]:
        thresholds = torch.zeros(len(self.labels))
        # Per class thresholds can be set here
        # Threshold sweeping
        for label in range(len(self.labels)):
            for threshold in torch.arange(0.1, 0.9, 0.1):
                metrics = MultiLabelMetrics(threshold).compute(prediction, expected)
                # Example: maximize F1 score
                if metrics["f1"] > thresholds[label]:
                    thresholds[label] = threshold

        return MultiLabelMetrics(0.5).compute(prediction, expected)
