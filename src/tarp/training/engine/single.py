import contextlib
from typing import ContextManager, cast, final

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


@final
class SingleDeviceEngine[ModelT: nn.Module]:
    def __init__(
        self,
        model: ModelT,
        device_idx: int = 0,
        mixed_precision: bool = True,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
    ):
        self._device = torch.device(
            f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu"
        )

        self._model = model.to(self._device)

        self.mixed_precision_dtype = mixed_precision_dtype
        self.mixed_precision = mixed_precision

        # Guard rails: GradScaler is only active for float16 computations.
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
    def is_rank_zero(self) -> bool:
        return True

    def autocast(self) -> ContextManager[object]:
        device_type = "cuda" if self._device.type == "cuda" else "cpu"
        return torch.amp.autocast(
            device_type=device_type,
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision,
        )

    def no_sync(self) -> ContextManager[object]:
        return contextlib.nullcontext()

    def zero_gradients(self) -> None:
        self._model.zero_grad(set_to_none=True)

    def backward_pass(self, loss: Tensor) -> None:
        self.scaler.scale(loss).backward()

    def step_optimizer(self, optimizer: Optimizer, clipping: float) -> bool:
        self.scaler.unscale_(optimizer)

        if clipping > 0.0:
            _ = torch.nn.utils.clip_grad_norm_(self._model.parameters(), clipping)

        initial_scale = self.scaler.get_scale()
        _ = self.scaler.step(optimizer)
        self.scaler.update()

        return self.scaler.get_scale() >= initial_scale
