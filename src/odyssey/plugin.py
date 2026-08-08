from dataclasses import dataclass

import torch

from .objective import Result
from .runtimes.core import RuntimeHandle


@dataclass(slots=True)
class State:
    """State tracker statically bound to a specific telemetry layout."""

    device: torch.device = torch.get_default_device()
    epoch_index: int = 0
    optimizer_step: int = 0
    local_accumulation_step: int = 0
    should_stop: bool = False


class Plugin[ResultT: Result]:
    def on_epoch_begin(
        self, state: State, is_training: bool, size: int, runtime: RuntimeHandle
    ) -> None:
        """
        Called at the start of each training epoch.
        """
        pass

    def on_batch_begin(
        self, state: State, is_training: bool, runtime: RuntimeHandle
    ) -> None:
        """
        Called before fetching a batch and running forward/backward passes.
        Use this for per-batch data-loading telemetry.
        """
        pass

    def on_batch_end(
        self, state: State, result: ResultT, is_training: bool, runtime: RuntimeHandle
    ) -> None:
        """
        Called after a single forward + backward pass.
        This fires whether we are accumulating gradients or updating weights.
        """
        pass

    def on_optimizer_step(self, state: State, runtime: RuntimeHandle) -> None:
        """
        Called when the optimizer successfully updates the model weights.
        This is where step-based learning rate schedulers should be called.
        """
        pass

    def on_epoch_end(
        self, state: State, is_training: bool, runtime: RuntimeHandle
    ) -> None:
        """
        Called at the end of each training epoch (e.g., validation, epoch-based schedulers).
        """
        pass
