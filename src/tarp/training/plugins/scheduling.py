from collections.abc import Iterable
from typing import final, override

from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from tarp.training.plugins.core import Plugin, State


@final
class BatchLearningScheduling[ResultT](Plugin[ResultT]):
    def __init__(
        self,
        schedulers: Iterable[LRScheduler],
    ) -> None:
        self.schedulers = schedulers

        for scheduler in self.schedulers:
            if isinstance(scheduler, ReduceLROnPlateau):
                raise ValueError(
                    "BatchLearningScheduling does not support ReduceLROnPlateau. Use EpochLearningScheduling instead."
                )

    @override
    def on_optimizer_step(self, state: State) -> None:
        for scheduler in self.schedulers:
            scheduler.step()


@final
class EpochLearningScheduling[ResultT](Plugin[ResultT]):
    def __init__(self, schedulers: Iterable[LRScheduler], metric_name: str) -> None:
        self.schedulers = schedulers
        self.metric_name = metric_name

    @override
    def on_epoch_end(self, state: State, is_training: bool) -> None:
        if not is_training:
            return

        for scheduler in self.schedulers:
            if isinstance(scheduler, ReduceLROnPlateau):
                if not state.metric_history:
                    raise ValueError(
                        "No metric history found. Make sure to log metrics before using ReduceLROnPlateau."
                    )
                metric_value = state.metric_history[-1][self.metric_name].item()
                scheduler.step(metric_value)
            else:
                scheduler.step()
