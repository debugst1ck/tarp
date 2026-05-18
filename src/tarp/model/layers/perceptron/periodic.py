import math
from typing import override

from torch import Tensor, nn

from tarp.functional.activations.periodic import sine


class SinusoidalFeedForward(nn.Module):
    """
    Sinusoidal Representation Network (SIREN) layer.
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        angular_frequency: float = 1.0,
        scale: float = 6.0,
        dropout: float = 0.0,
        is_first: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.angular_frequency = angular_frequency
        self.scale = scale
        self.is_first = is_first

        self.projection = nn.Linear(input_dimension, output_dimension, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.is_first:
            standard_deviation = 1.0 / self.projection.in_features
        else:
            standard_deviation = (
                math.sqrt(self.scale / self.projection.in_features)
                / self.angular_frequency
            )
        _ = nn.init.uniform_(
            self.projection.weight, -standard_deviation, standard_deviation
        )
        if self.projection.bias is not None:
            _ = nn.init.uniform_(
                self.projection.bias, -standard_deviation, standard_deviation
            )

    @override
    def forward(self, input: Tensor) -> Tensor:
        # Calls the functional sine calculation directly
        projected = self.projection(input)
        activated = sine(projected, angular_frequency=self.angular_frequency)
        return self.dropout(activated)
