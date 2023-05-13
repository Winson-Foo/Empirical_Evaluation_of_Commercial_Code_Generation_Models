"""
Set configuration values of the codebase.
"""

import torch


class Config:
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    OMP_NUM_THREADS = '1'
    MKL_NUM_THREADS = '1'
    TORCH_NUM_THREADS = 1
    SEED = 42
    HUBER_K = 1.0
    EPSILON = 0.1
    DIR_CHECKPOINT = 'checkpoints'