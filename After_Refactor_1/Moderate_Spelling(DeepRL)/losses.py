# File: losses.py

import torch

def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

def batch_trace(input):
    i = torch.arange(input.size(-1), device=input.device)
    t = input[:, i, i].sum(-1).unsqueeze(-1).unsqueeze(-1)
    return t

def batch_diagonal(input):
    dims = input.size()
    dims = dims + dims[-1:]
    output = torch.zeros(dims, device=input.device)
    strides = [output.stride(i) for i in range(input.dim() - 1)]
    strides.append(output.size(-1) + 1)
    output.as_strided(input.size(), strides).copy_(input)
    return output