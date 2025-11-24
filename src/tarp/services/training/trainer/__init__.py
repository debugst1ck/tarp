from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset

from tarp.cli.logging import Console
from tarp.services.training.callbacks import Callback
from tarp.services.training.callbacks.monitoring import (
    EarlyStopping,
    LearningRateScheduler,
)
from tarp.services.training.context import TrainerContext
from tarp.services.training.loops.training import TrainingLoop
from tarp.services.training.loops.validation import ValidationLoop
from tarp.services.training.state import TrainerState


class Trainer(ABC):
    def __init__(
        self,
        model: nn.Module,
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
        accumulation_steps: int = 1,
        callbacks: list[Callback] = [
            EarlyStopping(5),
            LearningRateScheduler("validation_loss"),
        ],
        shared: dict = {},
        persistent_workers: bool = False,
    ):
        """
        Base Trainer class.

        :param model: The model to be trained.
        :param train_dataset: The training dataset.
        :param valid_dataset: The validation dataset.
        :param optimizer: The optimizer for training.
        :param scheduler: The learning rate scheduler.
        :param device: The device to run the training on.
        :param batch_size: The batch size for training and validation.
        :param epochs: The number of training epochs.
        :param max_grad_norm: The maximum gradient norm for clipping.
        :param num_workers: The number of worker threads for data loading.
        :param use_amp: Whether to use automatic mixed precision.
        :param accumulation_steps: Number of steps to accumulate gradients before updating.
        :param callbacks: List of callback instances for training events.
        :param shared: A shared dictionary for storing custom data across callbacks and training steps.
        """
        self.context = TrainerContext(
            TrainerState(
                model=model.to(device),
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                scaler=torch.amp.GradScaler(enabled=use_amp),
                epochs=epochs,
                accumulation_steps=accumulation_steps,
                use_amp=use_amp,
                gradient_clipping_threshold=max_grad_norm,
                shared=shared,
            )
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if persistent_workers else False,
            persistent_workers=persistent_workers,
        )
        self.validation_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if persistent_workers else False,
            persistent_workers=persistent_workers,
        )

        self.callbacks = callbacks

        self.training_loop = TrainingLoop(
            context=self.context,
            forward=self.training_step,
            backpropagation=self.backpropagation,
            optimization=self.optimization,
            callbacks=self.callbacks,
        )
        self.validation_loop = ValidationLoop(
            context=self.context,
            forward=self.validation_step,
            evaluation=self.compute_metrics,
            callbacks=self.callbacks,
        )

    def _execute_callbacks(self, hook_name: str, **kwargs):
        for callback in self.callbacks:
            hook = getattr(callback, hook_name, None)
            if callable(hook):
                hook(self.context, **kwargs)

    @abstractmethod
    def training_step(
        self, batch: dict[str, Any]
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Perform a single training step.

        :param batch: A batch of data from the DataLoader.
        :return tuple[Tensor, Optional[Tensor], Optional[Tensor]]: The computed loss for the batch, predictions, and ground truths.
        """
        raise NotImplementedError

    @torch.no_grad()
    @abstractmethod
    def validation_step(
        self, batch: dict[str, Any]
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Perform a single validation step.

        :param batch: A batch of data from the DataLoader.
        :return tuple[Tensor, Optional[Tensor], Optional[Tensor]]: The computed loss for the batch, predictions, and ground truths.
        """
        raise NotImplementedError

    def compute_metrics(
        self, prediction: list[Tensor], expected: list[Tensor]
    ) -> dict[str, float]:
        """
        Compute metrics given logits and true labels.

        :param list[Tensor] prediction: List of model predictions (logits).
        :param list[Tensor] expected: List of true labels.
        :return dict[str, float]: A dictionary of computed metrics.
        """
        return {}

    def backpropagation(self, loss: Tensor) -> None:
        self.context.scaler.scale(loss).backward()

    def optimization(self) -> None:
        self.context.scaler.unscale_(self.context.optimizer)
        if self.context.gradient_clipping_threshold > 0:
            torch.nn.utils.clip_grad_norm_(
                self.context.model.parameters(),
                self.context.gradient_clipping_threshold,
            )
        self.context.scaler.step(self.context.optimizer)
        self.context.scaler.update()
        self.context.optimizer.zero_grad()

    def fit(self) -> None:
        self._execute_callbacks(Callback.on_training_start.__name__)
        for epoch in range(self.context.epochs):
            Console.info(
                f"Starting epoch {epoch + 1}/{self.context.epochs} for {self.__class__.__name__}"
            )

            self._execute_callbacks(Callback.on_epoch_start.__name__)

            # Training phase
            training_metrics = self.training_loop.run(epoch, self.train_dataloader)

            self.context.record_current_history(training_metrics)

            # Validation phase
            validation_metrics = self.validation_loop.run(
                epoch, self.validation_dataloader
            )

            self.context.record_current_history(validation_metrics)

            for key, value in self.context.current_metrics.items():
                Console.debug(f"{key}: {value:.4f}")

            self._execute_callbacks(Callback.on_epoch_end.__name__)

            if self.context.should_stop():
                break

            # Increment epoch count
            self.context.increment_epoch()

        self._execute_callbacks(Callback.on_training_end.__name__)
