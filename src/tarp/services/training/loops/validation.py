from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from tarp.services.training.loops import Loop


class ValidationLoop(Loop):
    def step(
        self, batch: dict[str, Tensor], optimize: bool = True
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        with torch.amp.autocast(
            device_type=self.context.device.type,
            enabled=self.context.use_amp,
        ):
            loss, predictions, expected = self.forward(batch)
        return loss, predictions, expected

    def run(self, epoch: int, dataloader: DataLoader) -> dict[str, float]:
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
                loss, predictions, expected = self.step(batch, optimize=False)
                total_loss += loss.item()
                if predictions is not None:
                    all_predictions.append(predictions)
                if expected is not None:
                    all_expected.append(expected)
                total_loss += loss.item()
                loop.set_postfix(loss=f"{loss.item():.4f}")
        average_loss = total_loss / len(dataloader)
        with torch.no_grad():
            metrics = self.evaluation(all_predictions, all_expected)
        metrics["validation_loss"] = average_loss
        return metrics
