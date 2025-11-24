from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from tarp.services.training.state import TrainerState


class TrainerContext:
    def __init__(self, state: TrainerState):
        self.state = state

    @property
    def device(self) -> torch.device:
        return self.state.device

    @property
    def model(self) -> nn.Module:
        return self.state.model

    @property
    def optimizer(self) -> Optimizer:
        return self.state.optimizer

    @property
    def scheduler(self) -> Optional[LRScheduler]:
        return self.state.scheduler

    @property
    def scaler(self) -> Optional[torch.amp.GradScaler]:
        return self.state.scaler

    def request_stop(self):
        self.state.stop_training = True

    def should_stop(self) -> bool:
        return self.state.stop_training

    def increment_epoch(self):
        self.state.epoch += 1

    @property
    def epoch(self) -> int:
        return self.state.epoch

    @property
    def accumulation_steps(self) -> int:
        return self.state.accumulation_steps

    @property
    def use_amp(self) -> bool:
        return self.state.use_amp

    @property
    def gradient_clipping_threshold(self) -> float:
        return self.state.gradient_clipping_threshold

    @property
    def epochs(self) -> int:
        return self.state.epochs

    def record_current_history(self, metrics: dict[str, float]):
        self.state.history[self.epoch].update(metrics)

    @property
    def current_metrics(self) -> dict[str, float]:
        return self.state.history[self.epoch]

    @property
    def shared(self) -> dict:
        return self.state.shared
