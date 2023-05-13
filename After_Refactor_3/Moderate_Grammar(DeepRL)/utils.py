# utils.py
import numpy as np
import torch

def range_tensor(n):
    return torch.arange(n).long()

def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = torch.tensor(x, dtype=torch.float32)
    return x.cuda() if torch.cuda.is_available() else x

def to_np(t):
    return t.cpu().detach().numpy() if torch.cuda.is_available() else t.detach().numpy()