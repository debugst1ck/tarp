from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class TrainerState:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        device: torch.device,
        scaler: Optional[torch.amp.GradScaler] = None,
        epochs: int = 10,
        accumulation_steps: int = 1,
        use_amp: bool = True,
        gradient_clipping_threshold: float = 1.0,
        shared: dict = {},
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.device = device
        self.history: list[dict[str, float]] = [{} for _ in range(epochs)]
        self.epochs = epochs
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp
        self.gradient_clipping_threshold = gradient_clipping_threshold

        self.epoch = 0
        self.stop_training = False

        self.shared: dict = shared
