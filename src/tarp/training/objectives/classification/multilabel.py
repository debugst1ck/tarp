from dataclasses import dataclass
from typing import final, override

import torch
from torch import Tensor, nn

from tarp.model.tasks.classification import ClassificationModel
from tarp.training.objectives.core import Objective
from tarp.typed.batch import ClassificationBatch


@dataclass(frozen=True)
class ClassificationResults:
    loss: Tensor
    scores: Tensor
    labels: Tensor


@final
class MultiLabelClassificationObjective(
    Objective[ClassificationModel, ClassificationBatch, ClassificationResults]
):
    def __init__(self, criterion: nn.Module | None) -> None:
        super().__init__()
        self.criterion = criterion or nn.BCEWithLogitsLoss()

    @override
    @torch.compile
    def forward_pass(
        self,
        model: ClassificationModel,
        batch: ClassificationBatch,
        device: torch.device,
    ) -> ClassificationResults:
        sequence = batch["sequence"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        scores, auxillary = model(sequence, attention_mask)
        loss = self.criterion(scores.view(-1, scores.size(-1)), labels.view(-1))

        if auxillary is not None:
            loss += auxillary

        return ClassificationResults(loss=loss, scores=scores, labels=labels)
