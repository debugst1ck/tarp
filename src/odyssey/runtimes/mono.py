import contextlib
from collections.abc import Iterable
from pathlib import Path
from threading import Thread
from typing import ContextManager, cast, final

import torch
from safetensors.torch import save_file, save_model
from torch import Tensor, nn
from torch.optim import Optimizer


@final
class MonoRuntime[ModelT: nn.Module]:
    def __init__(
        self,
        model: ModelT,
        mixed_precision: bool = True,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
    ):
        if torch.accelerator.is_available():
            self._device = cast(torch.device, torch.accelerator.current_accelerator())
        else:
            self._device = torch.get_default_device()

        self._model = model.to(self._device)

        self.mixed_precision_dtype = mixed_precision_dtype
        self.mixed_precision = mixed_precision

        # Running GradScaler with bfloat16 will waste cycles or raise warnings.
        use_scaler = mixed_precision and mixed_precision_dtype == torch.float16
        self.scaler = torch.amp.GradScaler(enabled=use_scaler)

    @property
    def model(self) -> ModelT:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_main_process(self) -> bool:
        return True

    def autocast(self) -> ContextManager[object]:
        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision,
        )

    def no_sync(self) -> ContextManager[object]:
        return contextlib.nullcontext()

    def zero_gradients(self, optimizers: Iterable[Optimizer]) -> None:
        for optimizer in optimizers:
            optimizer.zero_grad()

    def backward_pass(self, loss: Tensor) -> None:
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step_optimizers(self, optimizers: Iterable[Optimizer], clipping: float) -> bool:
        if self.scaler.is_enabled():
            for optimizer in optimizers:
                self.scaler.unscale_(optimizer)

        if clipping > 0.0:
            _ = torch.nn.utils.clip_grad_norm_(self._model.parameters(), clipping)

        initial_scale = self.scaler.get_scale()
        for optimizer in optimizers:
            if self.scaler.is_enabled():
                _ = self.scaler.step(optimizer)
            else:
                optimizer.step()

        if self.scaler.is_enabled():
            self.scaler.update()

        return self.scaler.get_scale() >= initial_scale

    def synchronize(self) -> None:
        torch.accelerator.synchronize(self.device)

    def checkpoint(self, path: Path, asynchronously: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        if asynchronously:
            self.synchronize()
            state_snapshot = {
                k: v.cpu().detach().clone().contiguous()
                for k, v in self.model.state_dict().items()
            }
            thread = Thread(
                target=save_file, args=(state_snapshot, path.as_posix()), daemon=True
            )
            thread.start()
        else:
            save_model(self.model, path.as_posix())
