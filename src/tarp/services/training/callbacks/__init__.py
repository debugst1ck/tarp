from tarp.services.training.context import TrainerContext


class Callback:
    def on_epoch_end(self, context: TrainerContext, **kwargs) -> None:
        pass

    def on_training_end(self, context: TrainerContext, **kwargs) -> None:
        pass

    def on_training_start(self, context: TrainerContext, **kwargs) -> None:
        pass

    def on_epoch_start(self, context: TrainerContext, **kwargs) -> None:
        pass

    def on_train_batch_end(self, context: TrainerContext, **kwargs) -> None:
        pass

    def on_train_batch_start(self, context: TrainerContext, **kwargs) -> None:
        pass

    def on_validation_batch_end(self, context: TrainerContext, **kwargs) -> None:
        pass

    def on_validation_batch_start(self, context: TrainerContext, **kwargs) -> None:
        pass
