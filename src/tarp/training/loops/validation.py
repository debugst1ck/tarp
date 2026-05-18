from typing import override

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tarp.training.callbacks.core import Callback
from tarp.training.loops.core import Loop
from tarp.typed.data import BatchT
from tarp.typed.training import PredictionT, TargetT


class ValidationLoop(Loop[BatchT, PredictionT, TargetT]):
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
            loss, predictions, expected = self.forward(batch, batch_index)
        return loss, predictions, expected

    @override
    def manual_step(
        self, batch: BatchT, batch_index: int, total_steps: int
    ) -> tuple[Tensor, PredictionT | None, TargetT | None]:
        self._execute_callbacks(Callback.on_validation_batch_start.__name__)
        with torch.no_grad():
            loss, predictions, expected = self.step(batch, batch_index, optimize=False)
        self._execute_callbacks(Callback.on_validation_batch_end.__name__)
        return loss, predictions, expected

    @override
    def run(self, epoch: int, dataloader: DataLoader[BatchT]) -> dict[str, float]:
        _ = self.context.model.eval()
        total_loss = 0.0
        all_expected: list[TargetT] = []
        all_predictions: list[PredictionT] = []
        loop: tqdm[BatchT] = tqdm(
            dataloader,
            desc=f"Validating [{epoch + 1}/{self.context.epochs}]",
            unit="batch",
            colour="red",
        )
        with torch.no_grad():
            for step, batch in enumerate(loop):
                loss, predictions, expected = self.manual_step(
                    batch,
                    batch_index=step + (epoch * len(dataloader)),
                    total_steps=len(dataloader),
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
