import os
from collections.abc import Iterable
from typing import ContextManager, final

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

from tarp.cli.core import Console


@final
class DistributedDataParallelEngine[ModelT: nn.Module]:
    def __init__(
        self,
        model: ModelT,
        mixed_precision: bool = True,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        find_unused_parameters: bool = False,
    ):
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self._global_rank = dist.get_rank() if dist.is_initialized() else 0

        self._device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )

        moved = model.to(self._device)

        if dist.is_initialized():
            self._model = DDP(
                moved,
                device_ids=[local_rank] if self._device.type == "cuda" else None,
                find_unused_parameters=find_unused_parameters,
            )
        else:
            raise RuntimeError(
                "DistributedDataParallelEngine requires torch.distributed to be initialized."
            )

        self.mixed_precision_dtype = mixed_precision_dtype
        self.mixed_precision = mixed_precision

        # Guard rails: GradScaler is only active for float16 computations.
        use_scaler = mixed_precision and mixed_precision_dtype == torch.float16
        if use_scaler and self.is_rank_zero:
            Console.warning(
                "GradScaler is enabled for mixed precision training with float16."
            )
        self.scaler = torch.amp.GradScaler(enabled=use_scaler)

    @property
    def model(self) -> DDP:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_rank_zero(self) -> bool:
        # Ensures console logging / progress bars print exclusively on the master branch
        return self._global_rank == 0

    def autocast(self) -> ContextManager[object]:
        device_type = "cuda" if self._device.type == "cuda" else "cpu"
        return torch.amp.autocast(
            device_type=device_type,
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision,
        )

    def no_sync(self) -> ContextManager[object]:
        return self.model.no_sync()

    def zero_gradients(self, optimizers: Iterable[Optimizer]) -> None:
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)

    def backward_pass(self, loss: Tensor) -> None:
        self.scaler.scale(loss).backward()

    def step_optimizers(self, optimizers: Iterable[Optimizer], clipping: float) -> bool:
        for optimizer in optimizers:
            self.scaler.unscale_(optimizer)

        if clipping > 0.0:
            _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clipping)

        initial_scale = self.scaler.get_scale()
        for optimizer in optimizers:
            _ = self.scaler.step(optimizer)
        self.scaler.update()

        return self.scaler.get_scale() >= initial_scale

    def barrier(self) -> None:
        if dist.is_initialized():
            dist.barrier()
