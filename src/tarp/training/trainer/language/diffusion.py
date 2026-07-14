from collections.abc import Sequence
from typing import final, override

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import Metric
from torchmetrics.text import Perplexity

from tarp.data.datasets.language.diffusion import CosineDiffusionMaskingDataset
from tarp.model.tasks.language import LanguageModel
from tarp.training.callbacks.core import Callback
from tarp.training.trainer.core import Trainer
from tarp.typed.batch import DiffusionBatch


@final
class DiffusionLanguageModelTrainer(
    Trainer[LanguageModel, DiffusionBatch, Tensor, Tensor]
):
    def __init__(
        self,
        model: LanguageModel,
        training_dataset: CosineDiffusionMaskingDataset,
        validation_dataset: CosineDiffusionMaskingDataset,
        optimizer: Optimizer,
        device: torch.device,
        criterion: nn.Module,
        scheduler: LRScheduler | None = None,
        batch_size: int = 32,
        epochs: int = 10,
        metrics: Sequence[Metric] = (Perplexity(ignore_index=-100),),
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
        self.vocabulary_size = training_dataset.tokenizer.vocabulary_size
        self.true_vocabulary_size = self.vocabulary_size - len(
            training_dataset.tokenizer.special_tokens_and_ids
        )

    @override
    def training_forward(
        self, batch: DiffusionBatch, batch_index: int
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        sequence = batch["sequence"].to(self.context.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(
            self.context.device, non_blocking=True
        )
        truth = batch["truth"].to(self.context.device, non_blocking=True)

        scores, auxillary = self.context.model(sequence, attention_mask=attention_mask)

        loss = self.criterion(
            scores.reshape(-1, self.vocabulary_size), truth.reshape(-1)
        )

        if auxillary is not None:
            loss += auxillary

        return loss, scores.detach().cpu(), truth.detach().cpu()

    @override
    def validation_forward(
        self, batch: DiffusionBatch, batch_index: int
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        sequence = batch["sequence"].to(self.context.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(
            self.context.device, non_blocking=True
        )
        truth = batch["truth"].to(self.context.device, non_blocking=True)
        scores, auxillary = self.context.model(sequence, attention_mask=attention_mask)
        loss = self.criterion(
            scores.reshape(-1, self.vocabulary_size), truth.reshape(-1)
        )
        if auxillary is not None:
            loss += auxillary

        return loss, scores.detach().cpu(), truth.detach().cpu()
