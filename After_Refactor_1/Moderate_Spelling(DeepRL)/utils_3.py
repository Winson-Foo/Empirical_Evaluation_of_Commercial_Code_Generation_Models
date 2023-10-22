# File: utils.py

import numpy as np
import torch

def to_device(x):
    if isinstance(x, torch.Tensor):
        return x.to(Config.DEVICE)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(Config.DEVICE)
    else:
        raise TypeError('Unsupported type: {}'.format(type(x)))

def epsilon_greedy(epsilon, x):
    if len(x.shape) == 1:
        return np.random.randint(len(x)) if np.random.rand() < epsilon else np.argmax(x)
    elif len(x.shape) == 2:
        random_actions = np.random.randint(x.shape[1], size=x.shape[0])
        greedy_actions = np.argmax(x, axis=-1)
        dice = np.random.rand(x.shape[0])
        return np.where(dice < epsilon, random_actions, greedy_actions)
    else:
        raise ValueError('Unsupported shape: {}'.format(x.shape))

def random_seed(seed=None):
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(int(1e6)))

def set_one_thread():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)