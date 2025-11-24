from typing import Optional

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from tarp.model.finetuning.language import LanguageModel
from tarp.services.datasets.language.masked import MaskedLanguageModelDataset
from tarp.services.training.trainer import Trainer


class MaskedLanguageModelTrainer(Trainer):
    def __init__(
        self,
        model: LanguageModel,
        train_dataset: MaskedLanguageModelDataset,
        valid_dataset: MaskedLanguageModelDataset,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        device: torch.device,
        vocabulary_size: int,
        batch_size: int = 32,
        epochs: int = 10,
        max_grad_norm: float = 1.0,
        num_workers: int = 0,
        use_amp: bool = True,
        accumulation_steps: int = 1,
        persistent_workers: bool = False,
        criterion: Optional[nn.Module] = None,
    ):
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            self.criterion = criterion
        self.criterion = self.criterion.to(device)

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
        self.vocab_size = vocabulary_size

    def training_step(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        sequence = batch["sequence"].to(self.context.device)
        attention_mask = batch["attention_mask"].to(self.context.device)
        truth = batch["truth"].to(self.context.device)

        # Model forward
        outputs = self.context.model(sequence, attention_mask=attention_mask)
        # Expect model to output logits of shape [batch, seq_len, vocab_size]
        logits = outputs if isinstance(outputs, Tensor) else outputs["logits"]

        # Flatten for loss
        loss = self.criterion(logits.view(-1, self.vocab_size), truth.view(-1))
        return loss, logits.detach().cpu(), truth.detach().cpu()

    @torch.no_grad()
    def validation_step(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        sequence = batch["sequence"].to(self.context.device)
        attention_mask = batch["attention_mask"].to(self.context.device)
        truth = batch["truth"].to(self.context.device)

        outputs = self.context.model(sequence, attention_mask=attention_mask)
        logits = outputs if isinstance(outputs, Tensor) else outputs["logits"]

        loss = self.criterion(logits.view(-1, self.vocab_size), truth.view(-1))
        return loss, logits.detach().cpu(), truth.detach().cpu()

    def compute_metrics(
        self, prediction: list[Tensor], expected: list[Tensor], topk: int = 5
    ) -> dict[str, float]:
        correct = 0
        total = 0
        topk_correct = 0
        # How can we not concatenate here?
        for logits, truth in zip(prediction, expected):
            # mask invalid tokens
            mask = truth != -100
            if mask.sum() == 0:
                continue

            # masked accuracy
            preds = logits.argmax(dim=-1)
            correct += (preds[mask] == truth[mask]).sum().item()
            total += mask.sum().item()

            if topk > 1:
                # get top-k indices along vocab dimension
                topk_preds = logits.topk(topk, dim=-1).indices  # [batch, seq_len, topk]

                # select masked positions
                topk_preds_masked = topk_preds[mask]  # [valid_tokens, topk]
                truth_masked = truth[mask].unsqueeze(-1)  # [valid_tokens, 1]

                # check if truth is in top-k predictions
                topk_correct += (
                    topk_preds_masked.eq(truth_masked).any(dim=-1).sum().item()
                )

        accuracy = correct / total if total > 0 else 0.0
        metrics = {"masked_accuracy": accuracy}
        if topk > 1:
            metrics[f"top_{topk}_accuracy"] = topk_correct / total if total > 0 else 0.0

        return metrics
