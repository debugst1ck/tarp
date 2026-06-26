from typing import override

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tarp.training.callbacks.core import Callback
from tarp.training.loops.core import Loop
from tarp.typed.data import BatchT
from tarp.typed.training import PredictionT, TargetT


class TrainingLoop(Loop[BatchT, PredictionT, TargetT]):
    @override
    def step(
        self,
        batch: BatchT,
        batch_index: int,
        optimize: bool = True,
    ) -> tuple[Tensor, PredictionT | None, TargetT | None]:
        with torch.amp.autocast_mode.autocast(
            device_type=self.context.device.type,
            enabled=self.context.is_mixed_precision,
            dtype=self.context.mixed_precision_dtype,
        ):
            raw_loss, prediction, expected = self.forward(batch, batch_index)
            loss = raw_loss / self.context.accumulation_steps
        self.backpropagation(loss)
        if optimize:
            stepped = self.optimization()
            if stepped:
                self._execute_callbacks(Callback.after_optimizer_step.__name__)
        return raw_loss, prediction, expected

    @override
    def manual_step(
        self, batch: BatchT, batch_index: int, total_steps: int
    ) -> tuple[Tensor, PredictionT | None, TargetT | None]:
        self._execute_callbacks(Callback.on_train_batch_start.__name__)
        accumulation_stop = (batch_index + 1) % self.context.accumulation_steps == 0
        is_last_step = (batch_index + 1) == total_steps
        optimize = accumulation_stop or is_last_step

        # Compute the step
        loss, prediction, expected = self.step(batch, batch_index, optimize)
        self._execute_callbacks(Callback.on_train_batch_end.__name__)

        return loss, prediction, expected

    @override
    def run(self, epoch: int, dataloader: DataLoader[BatchT]) -> dict[str, float]:
        _ = self.context.model.train()
        total_loss = 0.0
        loop: tqdm[BatchT] = tqdm(
            dataloader,
            desc=f"Training [{epoch + 1}/{self.context.epochs}]",
            unit="batch",
            colour="green",
        )
        for step, batch in enumerate(loop):
            loss, _, _ = self.manual_step(
                batch,
                batch_index=step + (epoch * len(dataloader)),
                total_steps=len(dataloader),
            )
            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        average_loss = total_loss / len(dataloader)
        return {"training_loss": average_loss}
