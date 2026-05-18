from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Generic

from torch import Tensor
from torch.utils.data import DataLoader

from tarp.training.callbacks.core import Callback
from tarp.training.context import TrainerContext
from tarp.typed.data import BatchT
from tarp.typed.training import ModelT, PredictionT, TargetT


class Loop(ABC, Generic[BatchT, PredictionT, TargetT]):
    def __init__(
        self,
        context: TrainerContext[ModelT],
        forward: Callable[
            [BatchT, int], tuple[Tensor, PredictionT | None, TargetT | None]
        ],
        evaluation: Callable[
            [Sequence[PredictionT], Sequence[TargetT]], Mapping[str, float]
        ] = lambda prediction, expected: {},
        backpropagation: Callable[[Tensor], None] = lambda loss: None,
        optimization: Callable[[], bool] = lambda: True,
        callbacks: Sequence[Callback] = (),
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

    def _execute_callbacks(self, hook_name: str):
        for callback in self.callbacks:
            hook = getattr(callback, hook_name, None)
            if callable(hook):
                _ = hook(self.context)

    @abstractmethod
    def run(self, epoch: int, dataloader: DataLoader[BatchT]) -> Mapping[str, float]:
        raise NotImplementedError

    @abstractmethod
    def step(
        self, batch: BatchT, batch_index: int, optimize: bool = True
    ) -> tuple[Tensor, PredictionT | None, TargetT | None]:
        raise NotImplementedError

    @abstractmethod
    def manual_step(
        self, batch: BatchT, batch_index: int, total_steps: int
    ) -> tuple[Tensor, PredictionT | None, TargetT | None]:
        raise NotImplementedError
