from pathlib import Path
from typing import final, override

from odyssey import Plugin, Result, RuntimeHandle, State


@final
class CheckpointOnEnd[ResultT: Result](Plugin[ResultT]):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path

    @override
    def on_epoch_end(
        self, state: State, is_training: bool, runtime: RuntimeHandle
    ) -> None:
        if is_training:
            runtime.checkpoint(self._path)
