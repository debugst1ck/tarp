from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from tarp.services.training.callbacks import Callback
from tarp.services.training.loops import Loop


class TrainingLoop(Loop):
    def step(
        self, batch: dict[str, Tensor], optimize: bool = True
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        with torch.amp.autocast_mode.autocast(
            device_type=self.context.device.type,
            enabled=self.context.use_amp,
        ):
            raw_loss, prediction, expected = self.forward(batch)
            loss = raw_loss / self.context.accumulation_steps
        self.backpropagation(loss)
        if optimize:
            self.optimization()
        return raw_loss, prediction, expected

    def manual_step(
        self, batch: dict[str, Tensor], step_index: int, total_steps: int
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        self._execute_callbacks(Callback.on_train_batch_start.__name__)
        # Determine if we should optimize on this step
        accumulation_stop = (step_index + 1) % self.context.accumulation_steps == 0
        is_last_step = (step_index + 1) == total_steps
        optimize = accumulation_stop or is_last_step
        # Compute the step
        loss, prediction, expected = self.step(batch, optimize=optimize)
        self._execute_callbacks(Callback.on_train_batch_end.__name__)
        return loss, prediction, expected

    def run(self, epoch: int, dataloader: DataLoader) -> dict[str, float]:
        self.context.model.train()
        total_loss = 0.0
        loop = tqdm(
            dataloader,
            desc=f"Training {epoch + 1}/{self.context.epochs}",
            unit="batch",
            colour="green",
        )
        for step, batch in enumerate(loop):
            loss, prediction, expected = self.manual_step(
                batch,
                step_index=step,
                total_steps=len(dataloader),
            )
            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        average_loss = total_loss / len(dataloader)
        return {"training_loss": average_loss}
