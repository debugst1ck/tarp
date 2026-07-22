from collections.abc import Iterable, Sized
from contextlib import nullcontext
from typing import Protocol, final

from torch import inference_mode, nn
from torch.optim import Optimizer
from tqdm.auto import tqdm

from tarp.training.engine.core import Engine
from tarp.training.objectives.core import Objective, Result
from tarp.training.plugins.core import Plugin, State
from tarp.typed.batch import SequenceBatch


class SizedIterable[T](Iterable[T], Sized, Protocol):
    pass


@final
class Orchestrator[ModelT: nn.Module, BatchT: SequenceBatch, ResultT: Result]:
    def __init__(
        self,
        engine: Engine[ModelT],
        objective: Objective[ModelT, BatchT, ResultT],
        optimizers: Iterable[Optimizer],
        plugins: Iterable[Plugin[ResultT]] | None = None,
        clipping: float = 1.0,
        accumulation_steps: int = 1,
    ) -> None:
        self.engine = engine
        self.objective = objective
        self.optimizers = optimizers
        self.plugins = plugins or ()
        self.clipping = clipping
        self.accumulation_steps = max(1, accumulation_steps)

    def _forward_pass(
        self,
        batch: BatchT,
    ) -> ResultT:
        with self.engine.autocast():
            return self.objective.forward_pass(
                self.engine.model,
                batch,
                self.engine.device,
            )

    def _iteration(
        self,
        batch: BatchT,
    ) -> ResultT:
        results = self._forward_pass(batch)
        scaled_loss = results.loss / self.accumulation_steps
        self.engine.backward_pass(scaled_loss)
        return results

    def run(
        self, dataloader: SizedIterable[BatchT], state: State, is_training: bool = True
    ) -> State:
        state.is_training = is_training

        for plugin in self.plugins:
            plugin.on_epoch_begin(state, is_training)

        _ = self.engine.model.train(is_training)
        self.engine.zero_gradients()

        total_batches = len(dataloader)

        description = "Training" if is_training else "Evaluating"
        progress_bar = tqdm(
            dataloader,
            desc=f"{description} Epoch {state.current_epoch}",
            total=total_batches,
            disable=not self.engine.is_rank_zero,
        )

        context = inference_mode() if not is_training else nullcontext()

        with context:
            for step_index, batch in enumerate(progress_bar, start=1):
                if state.should_stop:
                    break

                is_last_batch = step_index == total_batches
                is_step_boundary = False

                if is_training:
                    is_normal_boundary = (
                        state.accumulation_step + 1
                    ) >= self.accumulation_steps
                    is_step_boundary = is_normal_boundary or is_last_batch

                for plugin in self.plugins:
                    plugin.on_batch_begin(state, is_training)

                if is_training:
                    if not is_step_boundary:
                        with self.engine.no_sync():
                            result = self._iteration(batch)
                    else:
                        result = self._iteration(batch)

                    state.accumulation_step += 1
                else:
                    result = self._forward_pass(batch)

                # Synchronization point
                state.latest_loss = result.loss.detach().item()

                for plugin in self.plugins:
                    plugin.on_batch_end(state, result, is_training)

                if is_training and is_step_boundary:
                    optimized = self.engine.step_optimizers(
                        self.optimizers, self.clipping
                    )
                    state.accumulation_step = 0

                    if optimized:
                        state.global_optimizer_step += 1
                        for plugin in self.plugins:
                            plugin.on_optimizer_step(state)

                    self.engine.zero_gradients()

                if self.engine.is_rank_zero:
                    progress_bar.set_postfix(
                        loss=f"{state.latest_loss:.4f}",
                        step=state.global_optimizer_step,
                    )

        for plugin in self.plugins:
            plugin.on_epoch_end(state, is_training)

        return state
