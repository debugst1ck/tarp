from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from tarp.model.finetuning.metric.triplet import TripletMetricModel
from tarp.services.datasets.metric.triplet import MultiLabelOfflineTripletDataset
from tarp.services.training.trainer import Trainer


class OfflineTripletMetricTrainer(
    Trainer[dict[str, dict[str, Tensor]], tuple[Tensor, Tensor, Tensor], None]
):
    def __init__(
        self,
        model: TripletMetricModel,
        train_dataset: MultiLabelOfflineTripletDataset,
        valid_dataset: MultiLabelOfflineTripletDataset,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        device: torch.device,
        batch_size: int = 32,
        epochs: int = 10,
        max_grad_norm: float = 1.0,
        margin: float = 2.0,
        num_workers: int = 0,
        criterion: Optional[nn.Module] = None,
        accumulation_steps: int = 1,
        persistent_workers: bool = False,
    ):
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
            accumulation_steps=accumulation_steps,
            persistent_workers=persistent_workers,
        )

        # Loss function
        self.criterion = (
            criterion if criterion is not None else nn.TripletMarginLoss(margin=margin)
        )

    def _move_item_to_device(
        self, item: dict[str, Tensor]
    ) -> tuple[Tensor, Optional[Tensor]]:
        sequence = item["sequence"].to(self.context.device)
        mask = item.get("mask", None)
        if mask is not None:
            mask = mask.to(self.context.device)
        return sequence, mask

    def training_forward(
        self, batch: dict[str, dict[str, Tensor]], batch_index: int
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor], None]:
        anchor, anchor_mask = self._move_item_to_device(batch["anchor"])
        positive, positive_mask = self._move_item_to_device(batch["positive"])
        negative, negative_mask = self._move_item_to_device(batch["negative"])

        anchor_embedding, positive_embedding, negative_embedding = self.context.model(
            anchor,
            positive,
            negative,
            anchor_mask=anchor_mask,
            positive_mask=positive_mask,
            negative_mask=negative_mask,
        )

        loss = self.criterion(anchor_embedding, positive_embedding, negative_embedding)
        return (
            loss,
            (
                anchor_embedding.detach().cpu(),
                positive_embedding.detach().cpu(),
                negative_embedding.detach().cpu(),
            ),
            None,
        )

    def validation_step(
        self, batch: dict[str, dict[str, Tensor]], batch_index: int
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor], None]:
        anchor, anchor_mask = self._move_item_to_device(batch["anchor"])
        positive, positive_mask = self._move_item_to_device(batch["positive"])
        negative, negative_mask = self._move_item_to_device(batch["negative"])

        anchor_embedding, positive_embedding, negative_embedding = self.context.model(
            anchor,
            positive,
            negative,
            anchor_mask=anchor_mask,
            positive_mask=positive_mask,
            negative_mask=negative_mask,
        )

        anchor_embedding = F.normalize(anchor_embedding, dim=-1)
        positive_embedding = F.normalize(positive_embedding, dim=-1)
        negative_embedding = F.normalize(negative_embedding, dim=-1)

        loss = self.criterion(anchor_embedding, positive_embedding, negative_embedding)

        return (
            loss,
            (
                anchor_embedding.detach().cpu(),
                positive_embedding.detach().cpu(),
                negative_embedding.detach().cpu(),
            ),
            None,
        )

    def compute_metrics(
        self,
        prediction: Sequence[tuple[Tensor, Tensor, Tensor]],
        expected: Sequence[None],
    ) -> dict[str, float]:
        # Prediction is a list of tuples (anchor, positive, negative)

        anchors, positives, negatives = zip(*prediction)

        anchors = torch.cat(anchors, dim=0)
        positives = torch.cat(positives, dim=0)
        negatives = torch.cat(negatives, dim=0)

        # Compute distances
        positive_distances = torch.norm(anchors - positives, dim=1)
        negative_distances = torch.norm(anchors - negatives, dim=1)

        # Compute metrics
        margin: float = self.criterion.margin  # type: ignore

        # Accuracy: fraction of triplets where positive is closer than negative by at least margin
        accuracy = (
            ((negative_distances - positive_distances) > margin).float().mean().item()
        )

        return {
            "triplet_accuracy": accuracy,
            "mean_positive_distance": positive_distances.mean().item(),
            "mean_negative_distance": negative_distances.mean().item(),
        }
