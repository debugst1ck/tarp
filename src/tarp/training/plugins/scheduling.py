from collections.abc import Iterable
from typing import final, override

from torch import nn
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from odyssey import Plugin, Result, RuntimeHandle, State


@final
class BatchLearningScheduling[ResultT: Result](Plugin[ResultT]):
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
    def on_optimizer_step(self, state: State, runtime: RuntimeHandle) -> None:
        for scheduler in self.schedulers:
            scheduler.step()
