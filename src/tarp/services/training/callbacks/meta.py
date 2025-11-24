# Meta callbacks for training processes.

from tarp.services.training.callbacks import Callback
from tarp.services.training.context import TrainerContext


class MultiCallback(Callback):
    def __init__(
        self, contextual_callbacks: list[tuple[TrainerContext, Callback]]
    ) -> None:
        self.contextual_callbacks = contextual_callbacks

    def _dispatch(self, method_name: str, context: TrainerContext) -> None:
        for subcontext, callback in self.contextual_callbacks:
            method = getattr(callback, method_name, None)
            if callable(method):
                method(subcontext)

    # Explicitly override known callback hooks
    def on_train_start(self, context: TrainerContext, **kwargs):
        self._dispatch(Callback.on_training_start.__name__, context)

    def on_train_end(self, context: TrainerContext, **kwargs):
        self._dispatch(Callback.on_training_end.__name__, context)

    def on_epoch_start(self, context: TrainerContext, **kwargs):
        self._dispatch(Callback.on_epoch_start.__name__, context)

    def on_epoch_end(self, context: TrainerContext, **kwargs):
        self._dispatch(Callback.on_epoch_end.__name__, context)

    def on_train_batch_end(self, context: TrainerContext, **kwargs) -> None:
        self._dispatch(Callback.on_train_batch_end.__name__, context)

    def on_train_batch_start(self, context: TrainerContext, **kwargs) -> None:
        self._dispatch(Callback.on_train_batch_start.__name__, context)
