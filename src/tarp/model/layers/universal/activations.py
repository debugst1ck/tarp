import torch
from torch import Tensor, nn


class Sine(nn.Module):
    """
    Sine activation function with adjustable angular frequency.
    """

    def __init__(self, angular_frequency: float = 1.0) -> None:
        """
        :param angular_frequency: The angular frequency to scale the input.
        """
        super().__init__()
        self.angular_frequency = angular_frequency

    def forward(self, input: Tensor) -> Tensor:
        return torch.sin(self.angular_frequency * input)
