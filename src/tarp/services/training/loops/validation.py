from collections.abc import Mapping
from typing import Generic, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from tarp.services.training.callbacks import Callback
from tarp.services.training.loops import Loop
from tarp.typing.trainer import BatchT, PredT, TargetT


class ValidationLoop(Loop, Generic[BatchT, PredT, TargetT]):
    def step(
        self, batch: BatchT, batch_index: int, optimize: bool = True
    ) -> tuple[Tensor, Optional[PredT], Optional[TargetT]]:
        with torch.amp.autocast_mode.autocast(
            device_type=self.context.device.type,
            enabled=self.context.use_amp,
        ):
            loss, predictions, expected = self.forward(batch, batch_index)
        return loss, predictions, expected

    def manual_step(
        self, batch: BatchT, batch_index: int, total_steps: int
    ) -> tuple[Tensor, Optional[PredT], Optional[TargetT]]:
        self._execute_callbacks(Callback.on_validation_batch_start.__name__)
        with torch.no_grad():
            loss, predictions, expected = self.step(batch, batch_index, optimize=False)
        self._execute_callbacks(Callback.on_validation_batch_end.__name__)
        return loss, predictions, expected

    def run(self, epoch: int, dataloader: DataLoader) -> Mapping[str, float]:
        self.context.model.eval()
        total_loss = 0.0
        all_expected, all_predictions = [], []
        loop = tqdm(
            dataloader,
            desc=f"Validation {epoch + 1}/{self.context.epochs}",
            unit="batch",
            colour="red",
        )
        with torch.no_grad():
            for batch in loop:
                loss, predictions, expected = self.manual_step(
                    batch,
                    batch_index=0,
                    total_steps=0,
                )
                if predictions is not None:
                    all_predictions.append(predictions)
                if expected is not None:
                    all_expected.append(expected)
                total_loss += loss.item()
                loop.set_postfix(loss=f"{loss.item():.4f}")
        average_loss = total_loss / len(dataloader)
        with torch.no_grad():
            metrics = self.evaluation(all_predictions, all_expected)
        # Cast to dict to allow mutation
        metrics = dict(metrics)
        metrics["validation_loss"] = average_loss
        return metrics
