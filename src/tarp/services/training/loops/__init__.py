from abc import ABC, abstractmethod
from typing import Callable, Optional

from torch import Tensor
from torch.utils.data import DataLoader

from tarp.services.training.callbacks import Callback
from tarp.services.training.context import TrainerContext


class Loop(ABC):
    def __init__(
        self,
        context: TrainerContext,
        forward: Callable[
            [dict[str, Tensor]], tuple[Tensor, Optional[Tensor], Optional[Tensor]]
        ],
        evaluation: Callable[
            [list[Tensor], list[Tensor]], dict[str, float]
        ] = lambda prediction, expected: {},
        backpropagation: Callable[[Tensor], None] = lambda loss: None,
        optimization: Callable[[], None] = lambda: None,
        callbacks: list[Callback] = [],
    ):
        """
        Base class for training/evaluation loops.

        :param context: TrainerContext providing access to trainer state.
        :param iteration: Function to perform a single iteration (training/validation step).
        :param evaluation: Function to compute metrics given predictions and expected values.
        :param backpropagation: Function to perform backpropagation given a loss.
        :param optimization: Function to perform optimization step.
        """
        self.context = context
        self.forward = forward
        self.evaluation = evaluation
        self.backpropagation = backpropagation
        self.optimization = optimization
        self.callbacks = callbacks

    def _execute_callbacks(self, hook_name: str, **kwargs):
        for callback in self.callbacks:
            hook = getattr(callback, hook_name, None)
            if callable(hook):
                hook(self.context, **kwargs)

    @abstractmethod
    def run(self, epoch: int, dataloader: DataLoader) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def step(
        self, batch: dict[str, Tensor], optimize: bool = True
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def manual_step(
        self, batch: dict[str, Tensor], step_index: int, total_steps: int
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        raise NotImplementedError
