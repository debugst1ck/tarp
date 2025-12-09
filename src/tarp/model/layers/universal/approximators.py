import math

from torch import Tensor, nn

from tarp.model.layers.universal.activations import Sine


class SInusoidalREpresentationNetwork(nn.Module):
    """
    Sinusoidal Representation Network (SIREN) layer.
    Implements a single layer of a SIREN as described in the paper,
    - [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661) (Sitzmann et al., 2020)

    Each layer consists of a linear transformation followed by a sine activation function.
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
        """
        :param input_dimension: Input dimension I
        :param output_dimension: Output dimension O
        :param angular_frequency: Angular frequency ω
        :param scale: Scale factor for weight initialization
        :param dropout: Dropout rate
        :param is_first: Whether this is the first layer in the network
        :param bias: Whether to include a bias term in the linear layer
        """
        super().__init__()

        # Store parameters
        self.angular_frequency = angular_frequency
        self.scale = scale
        self.is_first = is_first

        # Create the linear layer, which will be initialized later
        self.projection = nn.Linear(input_dimension, output_dimension, bias=bias)
        self.activation = Sine(angular_frequency=angular_frequency)
        self.dropout = nn.Dropout(dropout)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.is_first:
            standard_deviation = 1.0 / self.projection.in_features
        else:
            standard_deviation = (
                math.sqrt(self.scale / self.projection.in_features)
                / self.angular_frequency
            )

        # Reinitialize weights
        # Weights are initialized from a uniform distribution U(-w_std, w_std)
        nn.init.uniform_(
            self.projection.weight, -standard_deviation, standard_deviation
        )
        # Reinitialize biases with the same standard deviation
        if self.projection.bias is not None:
            nn.init.uniform_(
                self.projection.bias, -standard_deviation, standard_deviation
            )

    def forward(self, input: Tensor) -> Tensor:
        """
        Equation = dropout(sin(ω * (W * x + b)))

        :param input: Input tensor of shape (..., I)
        :return: Output tensor of shape (..., O)
        """
        return self.dropout(self.activation(self.projection(input)))  # (..., O)
