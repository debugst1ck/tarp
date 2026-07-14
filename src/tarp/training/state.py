from typing import Generic, final

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from tarp.typed.training import ModelT


@final
class TrainerState(Generic[ModelT]):
    def __init__(
        self,
        model: ModelT,
        optimizer: Optimizer,
        device: torch.device,
        scheduler: LRScheduler | None = None,
        scaler: torch.amp.grad_scaler.GradScaler | None = None,
        epochs: int = 10,
        accumulation_steps: int = 1,
        distributed: bool = False,
        mixed_precision: bool = True,
        gradient_clipping_threshold: float = 1.0,
        shared: dict[str, object] = {},
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.device = device
        self.history: list[dict[str, float]] = [{} for _ in range(epochs)]
        self.epochs = epochs
        self.accumulation_steps = accumulation_steps
        self.distributed = distributed
        self.mixed_precision = mixed_precision
        self.gradient_clipping_threshold = gradient_clipping_threshold
        self.shared = shared

        self.epoch = 0
        self.stop_training = False
        self.paused = False
