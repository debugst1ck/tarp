from collections.abc import Collection, Iterable, Sized
from contextlib import nullcontext
from typing import Protocol, final

from torch import inference_mode, nn
from torch.optim import Optimizer

from .objective import Objective, Result
from .plugin import Plugin, State
from .runtimes.core import Runtime, RuntimeHandle


class BoundedIterable[T](Iterable[T], Sized, Protocol):
    pass


@final
class Orchestrator[ModelT: nn.Module, BatchT, ResultT: Result]:
    def __init__(
        self,
        runtime: Runtime[ModelT],
        objective: Objective[ModelT, BatchT, ResultT],
        optimizers: Iterable[Optimizer],
        plugins: Collection[Plugin[ResultT]] | None = None,
        clipping: float = 1.0,
        accumulation_steps: int = 1,
    ) -> None:
        self.runtime = runtime
        self.objective = objective
        self.optimizers = optimizers
        self.clipping = clipping
        self.accumulation_steps = max(1, accumulation_steps)
        self.runtime_handle = RuntimeHandle(runtime)

        self.plugins = plugins if plugins is not None else ()

    def _forward_pass(
        self,
        batch: BatchT,
    ) -> ResultT:
        with self.runtime.autocast():
            return self.objective.forward_pass(
                self.runtime.model,
                batch,
                self.runtime.device,
            )

    def _iteration(
        self,
        batch: BatchT,
    ) -> ResultT:
        results = self._forward_pass(batch)
        scaled_loss = results.loss / self.accumulation_steps
        self.runtime.backward_pass(scaled_loss)
        return results

    def run(
        self,
        dataloader: BoundedIterable[BatchT],
        state: State,
        is_training: bool = True,
    ) -> State:
        state.device = self.runtime.device
        total_batches = len(dataloader)

        for plugin in self.plugins:
            plugin.on_epoch_begin(
                state, is_training, total_batches, self.runtime_handle
            )

        _ = self.runtime.model.train(is_training)

        if is_training:
            self.runtime.zero_gradients(optimizers=self.optimizers)

        context = inference_mode() if not is_training else nullcontext()

        with context:
            for step_index, batch in enumerate(dataloader):
                if state.should_stop:
                    break

                is_last_batch = step_index == (total_batches - 1)
                is_step_boundary = False

                if is_training:
                    is_normal_boundary = (
                        state.local_accumulation_step + 1
                    ) == self.accumulation_steps
                    is_step_boundary = is_normal_boundary or is_last_batch

                for plugin in self.plugins:
                    plugin.on_batch_begin(state, is_training, self.runtime_handle)

                if is_training:
                    if not is_step_boundary:
                        with self.runtime.no_sync():
                            result = self._iteration(batch)
                    else:
                        result = self._iteration(batch)

                    state.local_accumulation_step += 1
                else:
                    result = self._forward_pass(batch)

                # Synchronization point
                for plugin in self.plugins:
                    plugin.on_batch_end(state, result, is_training, self.runtime_handle)

                if is_training and is_step_boundary:
                    optimized = self.runtime.step_optimizers(
                        self.optimizers, self.clipping
                    )
                    state.local_accumulation_step = 0

                    if optimized:
                        state.optimizer_step += 1
                        for plugin in self.plugins:
                            plugin.on_optimizer_step(state, self.runtime_handle)

                    self.runtime.zero_gradients(optimizers=self.optimizers)

        # Synchronize
        self.runtime.synchronize()

        for plugin in self.plugins:
            plugin.on_epoch_end(state, is_training, self.runtime_handle)

        # Update the epoch index and reset the accumulation step for the next epoch
        if is_training:
            state.epoch_index += 1
            state.local_accumulation_step = 0
        return state
