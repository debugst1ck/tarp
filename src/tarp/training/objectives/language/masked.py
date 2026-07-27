from dataclasses import dataclass
from typing import final, override

import torch
from torch import Tensor, nn

from tarp.model.tasks.language import LanguageModel
from tarp.training.objectives.core import Objective
from tarp.typed.batch import LanguageBatch


@dataclass
class LanguageModelResults:
    loss: Tensor
    scores: Tensor
    truth: Tensor


@final
class MaskedLanguageModelingObjective(
    Objective[LanguageModel, LanguageBatch, LanguageModelResults]
):
    def __init__(self, criterion: nn.Module | None) -> None:
        super().__init__()
        self.criterion = criterion or nn.CrossEntropyLoss()

    @override
    @torch.compile
    def forward_pass(
        self, model: LanguageModel, batch: LanguageBatch, device: torch.device
    ) -> LanguageModelResults:
        sequence = batch["sequence"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        truth = batch["truth"].to(device, non_blocking=True)

        scores, auxillary = model(sequence, attention_mask)

        loss = self.criterion(scores.view(-1, scores.size(-1)), truth.view(-1))
        if auxillary is not None:
            loss += auxillary

        return LanguageModelResults(loss=loss, scores=scores, truth=truth)
