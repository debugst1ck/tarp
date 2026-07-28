import contextlib
import os
from collections.abc import Callable, Iterable
from typing import Any, ContextManager, final

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.optim import Optimizer


@final
class FullyShardedDataParallel2NoSync:
    """Explicit context manager for FSDP2 gradient accumulation syncing."""

    def __init__(self, model: FSDPModule) -> None:
        self._model = model

    def __enter__(self) -> None:
        self._model.set_requires_gradient_sync(False)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._model.set_requires_gradient_sync(True)


@final
class FullyShardedDataParallelEngine[ModelT: nn.Module]:
    def __init__(
        self,
        model: ModelT,
        mixed_precision: bool = True,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        sharding_filter: Callable[[nn.Module], bool] | None = None,
    ):
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self._global_rank = dist.get_rank() if dist.is_initialized() else 0
        self._device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )

        # Move the base model to the GPU before sharding
        moved = model.to(self._device)

        if not dist.is_initialized():
            raise RuntimeError("FSDP2 requires torch.distributed initialization.")

        # Configure FSDP2 Mixed Precision Policy
        fsdp_kwargs = {}
        if mixed_precision:
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=mixed_precision_dtype,
                reduce_dtype=torch.float32,  # Best practice: keep reductions in fp32 for accuracy
            )

        # 1. Externally mark and shard submodules first
        if sharding_filter is not None:
            for module in moved.modules():
                if sharding_filter(module):
                    fully_shard(module, **fsdp_kwargs)

        # 2. Shard the root model wrapper (Required by FSDP2)
        self._model = fully_shard(moved, **fsdp_kwargs)

    @property
    def model(self) -> ModelT:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_rank_zero(self) -> bool:
        return self._global_rank == 0

    def autocast(self) -> ContextManager[object]:
        # FSDP2 manages internal parameter casting natively via `mp_policy`.
        # We return a nullcontext here to avoid redundant torch.amp.autocast overhead.
        return contextlib.nullcontext()

    def no_sync(self) -> ContextManager[object]:
        return FullyShardedDataParallel2NoSync(self._model)

    def zero_gradients(self, optimizers: Iterable[Optimizer]) -> None:
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)

    def backward_pass(self, loss: Tensor) -> None:
        loss.backward()

    def step_optimizers(self, optimizers: Iterable[Optimizer], clipping: float) -> bool:
        if clipping > 0.0:
            # torch.nn.utils.clip_grad_norm_ natively understands FSDP2 DTensors out-of-the-box
            _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clipping)
        for optimizer in optimizers:
            optimizer.step()
        return True

    def barrier(self) -> None:
        if dist.is_initialized():
            dist.barrier()
