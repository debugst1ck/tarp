from dataclasses import dataclass
from math import inf

import torch

from tarp.training.objectives.core import Result


@dataclass(slots=True)
class State:
    """State tracker statically bound to a specific telemetry layout."""

    device: torch.device = torch.get_default_device()
    current_epoch: int = 0
    current_epoch_step: int = 0
    current_accumulation_step: int = 0
    global_optimizer_step: int = 0
    accumulation_step: int = 0
    is_training: bool = True
    should_stop: bool = False
    latest_loss: float = inf


class Plugin[ResultT: Result]:
    def on_epoch_begin(self, state: State, is_training: bool) -> None:
        """
        Called at the start of each training epoch.
        """
        pass

    def on_epoch_end(self, state: State, is_training: bool) -> None:
        """
        Called at the end of each training epoch (e.g., validation, epoch-based schedulers).
        """
        pass

    def on_batch_begin(self, state: State, is_training: bool) -> None:
        """
        Called before fetching a batch and running forward/backward passes.
        Use this for per-batch data-loading telemetry.
        """
        pass

    def on_batch_end(
        self,
        state: State,
        result: ResultT,
        is_training: bool,
    ) -> None:
        """
        Called after a single forward + backward pass.
        This fires whether we are accumulating gradients or updating weights.
        """
        pass

    def on_optimizer_step(self, state: State) -> None:
        """
        Called when the optimizer successfully updates the model weights.
        This is where step-based learning rate schedulers should be called.
        """
        pass
