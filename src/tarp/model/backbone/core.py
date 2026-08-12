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
        positions: Tensor | None = None,
        mode: Literal["sequence", "pooled"],
    ) -> tuple[Tensor, Tensor | None]: ...
    @overload
    def encode(
        self,
        sequence_embeddings: Tensor,
        attention_mask: Tensor,
        *,
        positions: Tensor | None = None,
        mode: Literal["both"],
    ) -> tuple[Tensor, Tensor, Tensor | None]: ...
    @abstractmethod
    def encode(
        self,
        sequence_embeddings: Tensor,
        attention_mask: Tensor,
        *,
        positions: Tensor | None = None,
        mode: Literal["sequence", "pooled", "both"],
    ) -> tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor, Tensor | None]:
        """
        :param Tensor sequence_embeddings: [B, L, D]
        :param Tensor attention_mask: [B, L], 1 for valid positions, 0 for padding
        :param Tensor positions: [B, L], absolute positions for each token, optional
        :param str mode: "sequence" to return only sequence features [B, L, D], "pooled" to return only pooled features [B, D], "both" to return both
        :return tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor, Tensor | None]:
            If mode is "sequence": (sequence_features, auxiliary_loss)
            If mode is "pooled": (pooled_features, auxiliary_loss)
            If mode is "both": (sequence_features, pooled_features, auxiliary_loss)
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
        positions: Tensor | None = None,
        mode: Literal["sequence", "pooled", "both"] = "sequence",
    ) -> tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor, Tensor | None]:
        return self.encode(
            sequence_embeddings,
            attention_mask,
            positions=positions,
            mode=mode,
        )
