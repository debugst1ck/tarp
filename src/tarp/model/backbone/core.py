from abc import ABC, abstractmethod
from typing import Literal, Self, overload, override

from torch import Tensor, nn


class FrozenMixin(nn.Module, ABC):
    def freeze(self) -> Self:
        for param in self.parameters():
            param.requires_grad = False
        return self.eval()

    def unfreeze(self) -> Self:
        for param in self.parameters():
            param.requires_grad = True
        return self.train()


class Encoder(FrozenMixin, nn.Module, ABC):
    @overload
    def encode(
        self,
        sequence_embeddings: Tensor,
        attention_mask: Tensor,
        *,
        payload_mask: Tensor | None = None,
        positions: Tensor | None = None,
        mode: Literal["sequence"],
    ) -> Tensor: ...
    @overload
    def encode(
        self,
        sequence_embeddings: Tensor,
        attention_mask: Tensor,
        *,
        payload_mask: Tensor | None = None,
        positions: Tensor | None = None,
        mode: Literal["pooled"],
    ) -> Tensor: ...
    @overload
    def encode(
        self,
        sequence_embeddings: Tensor,
        attention_mask: Tensor,
        *,
        payload_mask: Tensor | None = None,
        positions: Tensor | None = None,
        mode: Literal["both"],
    ) -> tuple[Tensor, Tensor]: ...
    @abstractmethod
    def encode(
        self,
        sequence_embeddings: Tensor,
        attention_mask: Tensor,
        *,
        payload_mask: Tensor | None = None,
        positions: Tensor | None = None,
        mode: Literal["sequence", "pooled", "both"],
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        The core logic. Subclasses handle the 'mode' to prevent recomputation.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def encoding_size(self) -> int:
        raise NotImplementedError

    @override
    def forward(
        self,
        sequence_embeddings: Tensor,
        attention_mask: Tensor,
        *,
        payload_mask: Tensor | None = None,
        positions: Tensor | None = None,
        mode: Literal["sequence", "pooled", "both"] = "sequence",
    ) -> Tensor | tuple[Tensor, Tensor]:
        return self.encode(
            sequence_embeddings,
            attention_mask,
            payload_mask=payload_mask,
            positions=positions,
            mode=mode,
        )
