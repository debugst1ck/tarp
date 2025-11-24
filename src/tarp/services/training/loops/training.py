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
        with torch.amp.autocast(
            device_type=self.context.device.type,
            enabled=self.context.use_amp,
        ):
            raw_loss, prediction, expected = self.forward(batch)
            loss = raw_loss / self.context.accumulation_steps
        self.backpropagation(loss)
        if optimize:
            self.optimization()
        return raw_loss, prediction, expected

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
            self._execute_callbacks(Callback.on_train_batch_start.__name__)
            # Gradient accumulation step
            accumulation_stop = (step + 1) % self.context.accumulation_steps == 0
            is_last_step = (step + 1) == len(dataloader)
            loss, prediction, expected = self.step(
                batch, optimize=accumulation_stop or is_last_step
            )
            total_loss += loss.item()
            self._execute_callbacks(Callback.on_train_batch_end.__name__)
            loop.set_postfix(loss=f"{loss.item():.4f}")

        average_loss = total_loss / len(dataloader)
        return {"training_loss": average_loss}
