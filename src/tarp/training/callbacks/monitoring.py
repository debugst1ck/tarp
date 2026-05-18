from typing import final, override

from torch.optim.lr_scheduler import ReduceLROnPlateau

from tarp.cli.core import Console
from tarp.training.callbacks.core import Callback
from tarp.training.context import TrainerContext
from tarp.typed.training import ModelT


@final
class LearningRateScheduler(Callback):
    """
    Callback to step the learning rate scheduler at the end of each epoch.
    """

    @override
    def after_optimizer_step(self, context: TrainerContext[ModelT]) -> None:
        if context.scheduler is None:
            return
        if isinstance(context.scheduler, ReduceLROnPlateau):
            return
        context.scheduler.step()

    @override
    def on_training_start(self, context: TrainerContext[ModelT]) -> None:
        if context.scheduler is None:
            return
        if isinstance(context.scheduler, ReduceLROnPlateau):
            Console.warning(
                "Learning rate scheduler is ReduceLROnPlateau, not supported by LearningRateScheduler callback. Make sure to step the scheduler manually in your training loop based on the monitored metric."
            )
        else:
            Console.debug("Learning rate scheduler will step every optimizer step.")
