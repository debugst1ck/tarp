from typing import override

from torch import Tensor, nn

from tarp.functional.activations.gated import swiglu
from tarp.model.layers.perceptron.core import FeedForward


class SwishGatedLinearUnitFeedForward(FeedForward):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        bias: bool = True,
        hidden_dimension: int | None = None,
    ):
        super().__init__()
        hidden_dimension = hidden_dimension or output_dimension
        self.gate_and_content_projection = nn.Linear(
            input_dimension, 2 * hidden_dimension, bias=bias
        )
        self.output_projection = nn.Linear(
            hidden_dimension, output_dimension, bias=bias
        )

    @override
    def forward(self, features: Tensor) -> Tensor:
        """
        :param Tensor features: Input tensor of shape (..., input_dimension).
        :return Tensor: Output tensor of shape (..., output_dimension).
        """
        gate_and_content = self.gate_and_content_projection(features)
        gate, content = gate_and_content.chunk(2, dim=-1)
        return self.output_projection(swiglu(gate, content))
