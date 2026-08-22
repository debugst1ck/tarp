from dataclasses import dataclass
from typing import final, override

import torch
from odyssey import DefaultObjective
from torch import Tensor, nn

from tarp.model.tasks.language import LanguageModel
from tarp.typed.batch import LanguageBatch


@dataclass(frozen=True)
class LanguageModelResults:
    loss: Tensor
    scores: Tensor
    truth: Tensor


@final
class MaskedLanguageModelingObjective(
    DefaultObjective[LanguageModel, LanguageBatch, LanguageModelResults]
):
    def __init__(self, criterion: nn.Module | None = None) -> None:
        super().__init__()
        self.criterion = criterion or nn.CrossEntropyLoss()

    def preprocess(
        self, batch: LanguageBatch, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor]:
        sequence = batch["sequence"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        truth = batch["truth"].to(device, non_blocking=True)
        return sequence, attention_mask, truth

    def compute(
        self,
        model: LanguageModel,
        sequence: Tensor,
        attention_mask: Tensor,
        truth: Tensor,
    ) -> LanguageModelResults:
        scores, auxiliary = model(sequence, attention_mask)
        loss = self.criterion(scores.view(-1, scores.size(-1)), truth.view(-1))
        if auxiliary is not None:
            loss = loss + auxiliary
        return LanguageModelResults(loss=loss, scores=scores, truth=truth)

    @override
    def forward_pass(
        self, model: LanguageModel, *, batch: LanguageBatch, device: torch.device
    ) -> LanguageModelResults:
        sequence, attention_mask, truth = self.preprocess(batch, device)
        return self.compute(model, sequence, attention_mask, truth)
