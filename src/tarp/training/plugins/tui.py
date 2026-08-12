from typing import final, override

import torch
from torch import Tensor
from tqdm.auto import tqdm

from odyssey import Plugin, Result, RuntimeHandle, State


@final
class ProgressBar[ResultT: Result](Plugin[ResultT]):
    def __init__(self) -> None:
        super().__init__()
        self.progress_bar: tqdm | None = None

        self.accumulated_loss: Tensor | None = None
        self.batch_count: int = 0

    @override
    def on_epoch_begin(
        self, state: State, is_training: bool, size: int, runtime: RuntimeHandle
    ) -> None:
        self.accumulated_loss = torch.tensor(0.0, device=state.device)
        self.batch_count = 0

        if not runtime.is_main_process:
            return

        prefix = "Train" if is_training else "Infer"
        desc = f"{prefix} E{state.epoch_index + 1}"
        color = "green" if is_training else "blue"

        self.progress_bar = tqdm(
            total=size,
            desc=desc,
            unit="batch",
            colour=color,
            dynamic_ncols=True,
            disable=not runtime.is_main_process,
        )

    @override
    def on_batch_end(
        self, state: State, result: ResultT, is_training: bool, runtime: RuntimeHandle
    ) -> None:
        if self.accumulated_loss is not None:
            self.accumulated_loss += result.loss.detach()
            self.batch_count += 1

        if runtime.is_main_process and self.progress_bar is not None:
            self.progress_bar.update(1)

    @override
    def on_optimizer_step(self, state: State, runtime: RuntimeHandle) -> None:
        if self.batch_count == 0 or self.accumulated_loss is None:
            return

        step_loss = self.accumulated_loss / self.batch_count

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)

        if runtime.is_main_process and self.progress_bar is not None:
            self.progress_bar.set_postfix({"loss": f"{step_loss.item():.4f}"})

        # Reset buffers in-place without memory allocation
        _ = self.accumulated_loss.zero_()
        self.batch_count = 0

    @override
    def on_epoch_end(
        self, state: State, is_training: bool, runtime: RuntimeHandle
    ) -> None:
        if (
            not is_training
            and self.batch_count > 0
            and self.accumulated_loss is not None
        ):
            eval_loss = self.accumulated_loss / self.batch_count

            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    eval_loss, op=torch.distributed.ReduceOp.AVG
                )

            if runtime.is_main_process and self.progress_bar is not None:
                self.progress_bar.set_postfix({"loss": f"{eval_loss.item():.4f}"})

        if runtime.is_main_process and self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None

        self.accumulated_loss = None
        self.batch_count = 0
