from typing import final, override

import torch
from odyssey import (
    BatchTelemetry,
    ComputeHandle,
    EpochTelemetry,
    Plugin,
    Result,
    StepTelemetry,
)
from torch import Tensor
from torch.distributed import ReduceOp
from tqdm.auto import tqdm


@final
class ProgressBar[*ModelsTs, ObjectiveT, BatchT, ResultT: Result](
    Plugin[*ModelsTs, ObjectiveT, BatchT, ResultT]
):
    def __init__(self) -> None:
        super().__init__()
        self.progress_bar: tqdm | None = None
        self.accumulated_loss: Tensor | None = None
        self.batch_count: int = 0

    def _update_display(self, handle: ComputeHandle[*ModelsTs]) -> None:
        if self.batch_count == 0 or self.accumulated_loss is None:
            return
        average_loss = handle.reduce(
            self.accumulated_loss / self.batch_count, ReduceOp.AVG
        )
        if handle.is_main_process and self.progress_bar is not None:
            self.progress_bar.set_postfix({"loss": f"{average_loss.item():.4f}"})

        _ = self.accumulated_loss.zero_()
        self.batch_count = 0

    @override
    def on_epoch_begin(
        self, _telemetry: EpochTelemetry[*ModelsTs, ObjectiveT, BatchT, ResultT]
    ) -> None:
        is_training = _telemetry.is_training
        self.accumulated_loss = torch.tensor(0.0, device=_telemetry.handle.device)
        if not _telemetry.handle.is_main_process:
            return
        prefix = "Train" if is_training else "Infer"
        desc = f"{prefix} E{_telemetry.epoch_index + 1}"
        color = "green" if is_training else "blue"
        self.progress_bar = tqdm(
            total=_telemetry.total_batches,
            desc=desc,
            unit="batch",
            colour=color,
            dynamic_ncols=True,
            disable=not _telemetry.handle.is_main_process,
        )

    @override
    def on_batch_end(
        self,
        _telemetry: BatchTelemetry[*ModelsTs, ObjectiveT, BatchT, ResultT],
        _result: ResultT,
    ) -> None:
        if self.accumulated_loss is not None:
            self.accumulated_loss += _result.loss.detach()
            self.batch_count += 1

        if self.progress_bar is not None and _telemetry.handle.is_main_process:
            self.progress_bar.n = _telemetry.batch_index + 1
            self.progress_bar.refresh()

    @override
    def on_optimizer_step(
        self, _telemetry: StepTelemetry[*ModelsTs, ObjectiveT, BatchT, ResultT]
    ) -> None:
        self._update_display(_telemetry.handle)

    @override
    def on_epoch_end(
        self, _telemetry: EpochTelemetry[*ModelsTs, ObjectiveT, BatchT, ResultT]
    ) -> None:
        if not _telemetry.is_training:
            self._update_display(_telemetry.handle)
        if _telemetry.handle.is_main_process and self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None
        self.accumulated_loss = None
        self.batch_count = 0
