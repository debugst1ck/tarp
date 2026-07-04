from collections.abc import Mapping
from typing import Generic, final

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from tarp.training.state import TrainerState
from tarp.typed.training import ModelT


@final
class TrainerContext(Generic[ModelT]):
    def __init__(self, state: TrainerState[ModelT]):
        self.state = state

    @property
    def device(self) -> torch.device:
        return self.state.device

    @property
    def model(self) -> ModelT:
        return self.state.model

    @property
    def optimizer(self) -> Optimizer:
        return self.state.optimizer

    @property
    def scheduler(self) -> LRScheduler | None:
        return self.state.scheduler

    @property
    def scaler(self) -> torch.amp.grad_scaler.GradScaler | None:
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
    def is_mixed_precision(self) -> bool:
        return self.state.mixed_precision

    @property
    def gradient_clipping_threshold(self) -> float:
        return self.state.gradient_clipping_threshold

    @property
    def epochs(self) -> int:
        return self.state.epochs

    def record_current_history(self, metrics: Mapping[str, float]):
        self.state.history[self.epoch].update(metrics)

    @property
    def current_metrics(self) -> Mapping[str, float]:
        return self.state.history[self.epoch]

    @property
    def mixed_precision_dtype(self) -> torch.dtype:
        if self.is_mixed_precision:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            else:
                return torch.float16
        else:
            return torch.float32

    @property
    def shared(self) -> dict[str, object]:
        return self.state.shared
