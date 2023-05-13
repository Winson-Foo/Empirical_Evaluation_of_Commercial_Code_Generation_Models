# utils.py
import numpy as np
import torch

def to_tensor(x):
    return torch.from_numpy(np.asarray(x)).float()

def to_numpy(x):
    return x.data.cpu().numpy()

def epsilon_greedy(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(q_values))
    else:
        return np.argmax(q_values)