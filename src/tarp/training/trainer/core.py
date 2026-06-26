from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, Self

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchmetrics import Metric

from tarp.cli.core import Console
from tarp.data.datasets.core import SequenceDataset
from tarp.training.callbacks.core import Callback
from tarp.training.callbacks.monitoring import (
    LearningRateScheduler,
)
from tarp.training.context import TrainerContext
from tarp.training.loops.train import TrainingLoop
from tarp.training.loops.validation import ValidationLoop
from tarp.training.state import TrainerState
from tarp.typed.data import BatchT, RowT
from tarp.typed.training import ModelT, PredictionT, TargetT


class Trainer(ABC, Generic[ModelT, BatchT, PredictionT, TargetT]):
    def __init__(
        self,
        model: ModelT,
        training_dataset: SequenceDataset[RowT, BatchT],
        validation_dataset: SequenceDataset[RowT, BatchT],
        optimizer: Optimizer,
        device: torch.device,
        scheduler: LRScheduler | None = None,
        batch_size: int = 32,
        epochs: int = 10,
        metrics: Sequence[Metric] = (),
        gradient_clipping_threshold: float = 1.0,
        worker_count: int = 0,
        mixed_precision: bool = True,
        accumulation_steps: int = 1,
        persistent_workers: bool = True,
        callbacks: Sequence[Callback] | None = None,
        shared: dict[str, object] | None = None,
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

        self.context: TrainerContext[ModelT] = TrainerContext(
            TrainerState(
                model=model.to(device, non_blocking=True),
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                scaler=torch.amp.grad_scaler.GradScaler(enabled=mixed_precision),
                epochs=epochs,
                accumulation_steps=accumulation_steps,
                mixed_precision=mixed_precision,
                gradient_clipping_threshold=gradient_clipping_threshold,
                shared=shared if shared is not None else {},
            )
        )

        is_gpu = device.type == "cuda"

        self.train_dataloader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=worker_count,
            pin_memory=is_gpu,
            persistent_workers=is_gpu and persistent_workers,
            collate_fn=training_dataset.collate,
        )
        self.validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=worker_count,
            pin_memory=is_gpu,
            persistent_workers=is_gpu and persistent_workers,
            collate_fn=validation_dataset.collate,
        )

        self.callbacks = callbacks or (LearningRateScheduler(),)

        self.training_loop = TrainingLoop(
            context=self.context,
            forward=self.training_forward,
            metrics=metrics,
            backpropagation=self.backpropagation,
            optimization=self.optimization,
            callbacks=self.callbacks,
        )
        self.validation_loop = ValidationLoop(
            context=self.context,
            forward=self.validation_forward,
            metrics=metrics,
            callbacks=self.callbacks,
        )

    def _execute_callbacks(self, hook_name: str):
        for callback in self.callbacks:
            hook = getattr(callback, hook_name, None)
            if callable(hook):
                _ = hook(self.context)

    @abstractmethod
    def training_forward(
        self, batch: BatchT, batch_index: int
    ) -> tuple[Tensor, PredictionT | None, TargetT | None]:
        """
        Perform a single training step.

        :param batch: A batch of data from the DataLoader.
        :return tuple[Tensor, Optional[Tensor], Optional[Tensor]]: The computed loss for the batch, predictions, and ground truths.
        """
        raise NotImplementedError

    @torch.no_grad()
    @abstractmethod
    def validation_forward(
        self, batch: BatchT, batch_index: int
    ) -> tuple[Tensor, PredictionT | None, TargetT | None]:
        """
        Perform a single validation step.

        :param batch: A batch of data from the DataLoader.
        :return tuple[Tensor, Optional[Tensor], Optional[Tensor]]: The computed loss for the batch, predictions, and ground truths.
        """
        raise NotImplementedError

    def backpropagation(self, loss: Tensor) -> None:
        if self.context.scaler is not None:
            self.context.scaler.scale(loss).backward()
        else:
            loss.backward()

    def optimization(self) -> bool:
        if self.context.scaler is not None:
            self.context.scaler.unscale_(self.context.optimizer)

        if self.context.gradient_clipping_threshold > 0:
            _ = torch.nn.utils.clip_grad_norm_(
                self.context.model.parameters(),
                self.context.gradient_clipping_threshold,
            )

        if self.context.scaler is not None:
            old_scale = self.context.scaler.get_scale()
            _ = self.context.scaler.step(self.context.optimizer)
            self.context.scaler.update()

            # If scale decreased, the step was skipped due to an inf/nan value
            stepped = self.context.scaler.get_scale() >= old_scale
            if not stepped:
                Console.warning("Optimizer step skipped due to inf/nan gradients.")
        else:
            self.context.optimizer.step()
            stepped = True
        self.context.optimizer.zero_grad(set_to_none=True)
        return stepped

    def fit(self) -> Self:
        self._execute_callbacks(Callback.on_training_start.__name__)
        for epoch in range(self.context.epochs):
            Console.info(
                f"Starting epoch [{epoch + 1}/{self.context.epochs}] for {self.__class__.__name__}"
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
        return self
