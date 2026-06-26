from collections.abc import Sequence
from typing import final, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import Metric
from torchmetrics.text import Perplexity

from tarp.data.datasets.distillation.core import CrossDistillationDataset
from tarp.model.tasks.distillation import CrossLanguageDistillationModel
from tarp.model.tasks.language import LanguageModel
from tarp.training.callbacks.core import Callback
from tarp.training.trainer.core import Trainer
from tarp.typed.batch import DistillationBatch


@final
class CrossLanguageDistillationTrainer(
    Trainer[LanguageModel, DistillationBatch, Tensor, Tensor]
):
    def __init__(
        self,
        model: LanguageModel,
        distillation_model: CrossLanguageDistillationModel,
        training_dataset: CrossDistillationDataset,
        validation_dataset: CrossDistillationDataset,
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
        self.vocabulary_size = (
            training_dataset.student_dataset.tokenizer.vocabulary_size
        )
        self.true_vocabulary_size = self.vocabulary_size - len(
            training_dataset.tokenizer.special_tokens_and_ids
        )
        self.distillation_model = distillation_model.to(self.context.device)

    @override
    def training_forward(
        self, batch: DistillationBatch, batch_index: int
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        language_head = self.context.model.language_head

        distillation_model = self.distillation_model

        student_batch = batch["student"]  # MLM
        teacher_batch = batch["teacher"]  # Contrastive

        sequence = student_batch["sequence"].to(self.context.device, non_blocking=True)
        mask = student_batch["attention_mask"].to(
            self.context.device, non_blocking=True
        )
        truth = student_batch["truth"].to(self.context.device, non_blocking=True)

        teacher_sequence = teacher_batch["sequence"].to(
            self.context.device, non_blocking=True
        )
        teacher_mask = teacher_batch["attention_mask"].to(
            self.context.device, non_blocking=True
        )

        student_encoded, oracle, teacher_encoded, auxiliary = distillation_model(
            student_sequence=sequence,
            student_mask=mask,
            teacher_sequence=teacher_sequence,
            teacher_mask=teacher_mask,
        )

        # Compute the MLM loss
        scores = language_head(student_encoded)
        mlm_loss = self.criterion(
            scores.reshape(-1, self.vocabulary_size), truth.reshape(-1)
        )

        # MLM for oracle
        oracle_scores = language_head(oracle)
        mlm_oracle_loss = self.criterion(
            oracle_scores.reshape(-1, self.vocabulary_size), truth.reshape(-1)
        )

        mlm_positions = truth != -100
        temperature = 2.0

        kl_loss = F.kl_div(
            F.log_softmax(scores[mlm_positions] / temperature, dim=-1),
            F.softmax(oracle_scores.detach()[mlm_positions] / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature * temperature)

        loss = mlm_loss + mlm_oracle_loss + kl_loss
        if auxiliary is not None:
            loss += auxiliary

        if batch_index % self.context.accumulation_steps == 0:
            print(
                f"Batch {batch_index}: MLM Loss = {mlm_loss.item():.4f}, Oracle MLM Loss = {mlm_oracle_loss.item():.4f}, KL Loss = {kl_loss.item():.4f}"
            )

        return loss, scores.detach().cpu(), truth.detach().cpu()

    @override
    def validation_forward(
        self, batch: DistillationBatch, batch_index: int
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        language_head = self.context.model.language_head

        distillation_model = self.distillation_model

        student_batch = batch["student"]  # MLM
        teacher_batch = batch["teacher"]  # Contrastive

        sequence = student_batch["sequence"].to(self.context.device, non_blocking=True)
        mask = student_batch["attention_mask"].to(
            self.context.device, non_blocking=True
        )
        truth = student_batch["truth"].to(self.context.device, non_blocking=True)

        teacher_sequence = teacher_batch["sequence"].to(
            self.context.device, non_blocking=True
        )
        teacher_mask = teacher_batch["attention_mask"].to(
            self.context.device, non_blocking=True
        )

        student_encoded, oracle, teacher_encoded, auxiliary = distillation_model(
            student_sequence=sequence,
            student_mask=mask,
            teacher_sequence=teacher_sequence,
            teacher_mask=teacher_mask,
        )

        # Compute the MLM loss
        scores = language_head(student_encoded)
        mlm_loss = self.criterion(
            scores.reshape(-1, self.vocabulary_size), truth.reshape(-1)
        )

        # MLM for oracle
        oracle_scores = language_head(oracle)
        mlm_oracle_loss = self.criterion(
            oracle_scores.reshape(-1, self.vocabulary_size), truth.reshape(-1)
        )

        mlm_positions = truth != -100
        temperature = 2.0

        kl_loss = F.kl_div(
            F.log_softmax(scores[mlm_positions] / temperature, dim=-1),
            F.softmax(oracle_scores.detach()[mlm_positions] / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature * temperature)

        loss = mlm_loss + mlm_oracle_loss + kl_loss
        if auxiliary is not None:
            loss += auxiliary

        return loss, scores.detach().cpu(), oracle_scores.detach().cpu()
