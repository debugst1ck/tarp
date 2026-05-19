from collections.abc import Sequence
from typing import override

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from tarp.cli.core import Console
from tarp.data.datasets.language.masked import MaskedLanguageDataset
from tarp.functional.evaluation.classification import accuracy, top_k_accuracy
from tarp.model.heads.language import LanguageModel
from tarp.training.callbacks.core import Callback
from tarp.training.trainer.core import Trainer


class LanguageReconstructionTrainer(
    Trainer[LanguageModel, dict[str, Tensor], Tensor, Tensor]
):
    def __init__(
        self,
        model: LanguageModel,
        training_dataset: MaskedLanguageDataset,
        validation_dataset: MaskedLanguageDataset,
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
        self.vocabulary_size = training_dataset.tokenizer.vocabulary_size
        self.true_vocabulary_size = self.vocabulary_size - len(
            training_dataset.tokenizer.special_tokens_and_ids
        )

    @override
    def training_forward(
        self, batch: dict[str, Tensor], batch_index: int
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        sequence = batch["sequence"].to(self.context.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(
            self.context.device, non_blocking=True
        )
        truth = batch["truth"].to(self.context.device, non_blocking=True)
        # For pure payload it is (truth != -100) & attention_mask.bool()
        payload_mask = attention_mask
        original_sequence = torch.where(
            attention_mask.bool(),
            torch.where(truth != -100, truth, sequence),
            torch.full_like(sequence, -100),
        )

        scores, auxillary = self.context.model(
            sequence,
            attention_mask=attention_mask,
            payload_mask=payload_mask,
        )

        loss = self.criterion(
            scores.reshape(-1, self.vocabulary_size), truth.reshape(-1)
        )
        reconstruction_loss = self.criterion(
            scores.reshape(-1, self.vocabulary_size),
            original_sequence.reshape(-1),
        )

        loss += reconstruction_loss
        if auxillary is not None:
            loss += auxillary

        if batch_index % 100 == 0:
            Console.debug(
                f"Batch {batch_index}: Loss = {loss.item():.4f}, Reconstruction Loss = {reconstruction_loss.item():.4f}, Auxillary loss = {auxillary.item() if auxillary is not None else 0.0:.4f}"
            )

        return loss, scores.detach().cpu(), truth.detach().cpu()

    @override
    def validation_forward(
        self, batch: dict[str, Tensor], batch_index: int
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        sequence = batch["sequence"].to(self.context.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(
            self.context.device, non_blocking=True
        )
        truth = batch["truth"].to(self.context.device, non_blocking=True)
        payload_mask = attention_mask
        original_sequence = torch.where(
            attention_mask.bool(),
            torch.where(truth != -100, truth, sequence),
            torch.full_like(sequence, -100),
        )

        scores, auxillary = self.context.model(
            sequence,
            attention_mask=attention_mask,
            payload_mask=payload_mask,
        )

        loss = self.criterion(
            scores.reshape(-1, self.vocabulary_size), truth.reshape(-1)
        )
        reconstruction_loss = self.criterion(
            scores.reshape(-1, self.vocabulary_size),
            original_sequence.reshape(-1),
        )

        loss += reconstruction_loss
        if auxillary is not None:
            loss += auxillary

        return loss, scores.detach().cpu(), truth.detach().cpu()

    @override
    def compute_metrics(
        self, predictions: Sequence[Tensor], targets: Sequence[Tensor], top_k: int = 2
    ) -> dict[str, float]:
        correct = 0
        total = 0
        top_k_correct = 0
        for scores, truth in zip(predictions, targets):
            mask = truth != -100
            if mask.sum() == 0:
                continue
            prediction = scores.argmax(dim=-1)[mask]
            top_k_prediction = scores.topk(top_k, dim=-1).indices[mask]

            correct += (prediction == truth[mask]).sum().item()
            total += mask.sum().item()
            top_k_correct += (
                (top_k_prediction == truth[mask].unsqueeze(-1)).any(dim=-1).sum().item()
            )
        accuracy_metric = correct / total if total > 0 else 0.0
        top_k_accuracy_metric = top_k_correct / total if total > 0 else 0.0
        return {
            "accuracy": accuracy_metric,
            f"top_{top_k}_accuracy": top_k_accuracy_metric,
        }
