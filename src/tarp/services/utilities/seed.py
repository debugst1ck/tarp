import random

import numpy as np
import torch


def establish_random_seed(seed: int = 42) -> int:
    """
    Set the random seed for reproducibility across various libraries.

    :param int seed: The seed value to set. Default is 42, cause it's the answer to life, universe and everything.
    :return int: The seed value that was set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
