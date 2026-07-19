from typing import override

from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from tarp.training.plugins.core import Plugin, State


class BatchLearningScheduling[PredictionT, TargetT](Plugin[PredictionT, TargetT]):
    def __init__(
        self,
        scheduler: LRScheduler,
    ) -> None:
        self.scheduler = scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            raise ValueError(
                "BatchLearningScheduling does not support ReduceLROnPlateau. Use EpochLearningScheduling instead."
            )

    @override
    def on_optimizer_step(self, state: State) -> None:
        self.scheduler.step()


class EpochLearningScheduling[PredictionT, TargetT](Plugin[PredictionT, TargetT]):
    def __init__(self, scheduler: LRScheduler, metric_name: str) -> None:
        self.scheduler = scheduler
        self.metric_name = metric_name

    @override
    def on_epoch_end(self, state: State, is_training: bool) -> None:
        if not is_training:
            return

        if isinstance(self.scheduler, ReduceLROnPlateau):
            metric_value = state.metric_history[-1][self.metric_name].item()
            self.scheduler.step(metric_value)
        else:
            self.scheduler.step()
