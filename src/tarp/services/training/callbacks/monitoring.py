from torch.optim.lr_scheduler import ReduceLROnPlateau

from tarp.cli.logging import Console
from tarp.services.evaluation import Extremum
from tarp.services.training.callbacks import Callback
from tarp.services.training.context import TrainerContext


class EarlyStopping(Callback):
    def __init__(
        self,
        patience: int = 3,
        monitor_metric: str = "validation_loss",
        monitor_mode: Extremum = Extremum.MIN,
    ) -> None:
        self.patience = patience
        self.best_metric_value: float = (
            float("inf") if monitor_mode == Extremum.MIN else float("-inf")
        )
        self.counter = 0
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode

    def on_epoch_end(self, context: TrainerContext) -> None:
        current_value = context.state.history[context.epoch].get(self.monitor_metric)
        if current_value is None:
            return

        if self.best_metric_value is None or (
            (
                self.monitor_mode == Extremum.MIN
                and current_value < self.best_metric_value
            )
            or (
                self.monitor_mode == Extremum.MAX
                and current_value > self.best_metric_value
            )
        ):
            self.best_metric_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                Console.info("Early stopping triggered.")
                context.request_stop()

    def on_training_start(self, context: TrainerContext) -> None:
        Console.debug(f"Monitoring {self.monitor_metric} for early stopping.")


class LearningRateScheduler(Callback):
    """
    Callback to step the learning rate scheduler at the end of each epoch.
    """

    def __init__(self, monitor_metric: str) -> None:
        self.monitor_metric = monitor_metric

    def on_epoch_end(self, context: TrainerContext) -> None:
        if context.scheduler is None:
            return
        if isinstance(context.scheduler, ReduceLROnPlateau):
            current_value = context.state.history[context.epoch].get(
                self.monitor_metric
            )
            if current_value is not None:
                context.scheduler.step(current_value)
                if context.scheduler.num_bad_epochs == context.scheduler.patience:
                    Console.info(
                        f"Learning rate reduced to {context.scheduler._last_lr}."
                    )
        else:
            context.scheduler.step()

    def on_training_start(self, context, **kwargs):
        if context.scheduler is None:
            return
        if isinstance(context.scheduler, ReduceLROnPlateau):
            Console.debug(
                f"Learning rate scheduler will monitor {self.monitor_metric}."
            )
        else:
            Console.debug("Learning rate scheduler will step every epoch.")
