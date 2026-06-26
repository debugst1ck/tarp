from typing import override

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class JohnsonLindenstraussTransformation(nn.Module):
    """
    A linear layer that applies a random projection to reduce dimensionality while preserving distances, based on the Johnson-Lindenstrauss lemma.
    """

    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)

        projection = torch.randn(
            out_features, in_features, generator=generator
        ) / torch.sqrt(torch.tensor(out_features))

        self.projection: Tensor
        self.register_buffer("projection", projection, persistent=False)

    @override
    def forward(self, features: Tensor) -> Tensor:
        """
        Apply the random projection to the input features.

        :param features: A tensor of shape (batch_size, in_features).
        :return: A tensor of shape (batch_size, out_features) after projection.
        """
        return F.linear(features, self.projection)
