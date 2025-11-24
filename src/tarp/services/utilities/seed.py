# This file is for setting random seeds for reproducibility

import random
from typing import Optional

import numpy as np
import torch


def establish_random_seed(seed: Optional[int] = None) -> int:
    """
    Set the random seed for reproducibility across various libraries.

    Args:
        seed (Optional[int]): The seed value to set. If None, a random seed will be generated.
    Returns:
        int: The seed value that was set.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
