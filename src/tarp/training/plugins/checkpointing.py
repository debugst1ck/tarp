from pathlib import Path
from typing import final, override

from odyssey import (
    EpochTelemetry,
    Plugin,
)
from safetensors.torch import save_file


@final
class CheckpointOnEnd[*ModelsTs, ObjectiveT, BatchT, ResultT](
    Plugin[*ModelsTs, ObjectiveT, BatchT, ResultT]
):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path

    @override
    def on_epoch_end(
        self, _telemetry: EpochTelemetry[*ModelsTs, ObjectiveT, BatchT, ResultT]
    ) -> None:
        if _telemetry.is_training and _telemetry.handle.is_main_process:
            for state_dict in _telemetry.handle.state_dicts():
                save_file(state_dict, self._path)
