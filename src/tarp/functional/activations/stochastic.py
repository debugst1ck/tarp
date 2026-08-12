import torch
from torch import Tensor


def gumbel_softmax(
    scores: Tensor, temperature: float = 1.0, hard: bool = False, dimension: int = -1
):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.

    :param Tensor scores: Unnormalized log probabilities of shape (..., num_classes)
    :param float temperature: Non-negative scalar controlling the sharpness of the distribution. Lower temperatures yield more discrete distributions.
    :param bool hard: If True, the returned samples will be one-hot encoded, but gradients will still flow through the soft sample.
    :param int dimension: The dimension along which the softmax will be computed.
    :return: A tensor of the same shape as `scores` containing the sampled probabilities. If `hard=True`, the returned tensor will be one-hot encoded along the specified dimension.
    """
    # Sample Gumbel noise from the Gumbel distribution using the inverse transform sampling method
    gumbel_noise = -torch.empty_like(scores, device=scores.device).exponential_().log()
    # Apply temperature scaling and add Gumbel noise to the logits, then apply softmax to get the probabilities
    y_soft = ((scores + gumbel_noise) / temperature).softmax(dim=dimension)
    if hard:
        # Get one hot indices by taking the argmax of the soft sample
        index = y_soft.argmax(dim=dimension, keepdim=True)
        # Create a one-hot encoded tensor with the same shape as logits
        y_hard = torch.zeros_like(scores).scatter_(dimension, index, 1.0)
        # Straight through estimation
        return (y_hard - y_soft).detach() + y_soft
    else:
        return y_soft  # Return soft sample
