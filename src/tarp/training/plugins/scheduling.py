from collections.abc import Sequence
from typing import final, override

from odyssey import (
    Plugin,
    StepTelemetry,
)
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau


@final
class BatchLearningScheduling[*ModelsTs, ObjectiveT, BatchT, ResultT](
    Plugin[*ModelsTs, ObjectiveT, BatchT, ResultT]
):
    def __init__(self, schedulers: Sequence[LRScheduler]) -> None:
        super().__init__()
        self.schedulers = schedulers
        for scheduler in self.schedulers:
            if isinstance(scheduler, ReduceLROnPlateau):
                raise TypeError(
                    "BatchLearningScheduling does not support ReduceLROnPlateau. Use EpochLearningScheduling instead."
                )

    @override
    def on_optimizer_step(
        self, _telemetry: StepTelemetry[*ModelsTs, ObjectiveT, BatchT, ResultT]
    ) -> None:
        for scheduler in self.schedulers:
            scheduler.step()
