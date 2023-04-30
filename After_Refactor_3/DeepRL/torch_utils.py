import torch
import numpy as np


def set_random_seeds(seed: int) -> None:
    """
    Sets random seeds for Python, Pytorch, and NumPy libraries.

    :param seed: the random seed value to use.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def to_np(t: torch.Tensor) -> np.array:
    """
    Converts a Pytorch tensor to a NumPy array.

    :param t: the tensor to convert.
    :return: the tensor as a NumPy array.
    """
    return t.detach().cpu().numpy()