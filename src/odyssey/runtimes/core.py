from collections.abc import Iterable
from pathlib import Path
from typing import ContextManager, Protocol, final

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


class Runtime[ModelT: nn.Module](Protocol):
    @property
    def model(self) -> ModelT: ...

    @property
    def device(self) -> torch.device: ...

    @property
    def is_main_process(self) -> bool: ...

    def autocast(self) -> ContextManager[object]: ...
    def no_sync(self) -> ContextManager[object]: ...

    def zero_gradients(self, optimizers: Iterable[Optimizer]) -> None: ...
    def backward_pass(self, loss: Tensor) -> None: ...
    def step_optimizers(
        self, optimizers: Iterable[Optimizer], clipping: float
    ) -> bool: ...
    def synchronize(self) -> None: ...
    def checkpoint(self, path: Path, asynchronously: bool = False) -> None: ...


@final
class RuntimeHandle:
    def __init__(self, runtime: Runtime[nn.Module]) -> None:
        self._runtime = runtime

    def checkpoint(self, path: Path, asynchronously: bool = False) -> None:
        self._runtime.checkpoint(path, asynchronously)

    @property
    def is_main_process(self) -> bool:
        return self._runtime.is_main_process
