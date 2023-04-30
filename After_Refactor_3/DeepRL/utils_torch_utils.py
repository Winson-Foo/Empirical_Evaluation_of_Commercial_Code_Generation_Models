# -*- coding: utf-8 -*-
"""Utility functions for Reinforcement Learning"""

from typing import Union
import os
import torch
import numpy as np
from torch import Tensor
from torch.distributions import Normal, Categorical

from .config import Config


def select_device(gpu_id: int) -> None:
    """Selects whether to use CPU or GPU"""
    if cuda.is_available() and gpu_id >= 0:
        Config.DEVICE = torch.device('cuda:%d' % (gpu_id))
    else:
        Config.DEVICE = torch.device('cpu')


def to_tensor(x: Union[Tensor, np.ndarray, float, int]) -> Tensor:
    """Convert input to PyTorch tensor"""
    if isinstance(x, Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).to(Config.DEVICE)
    return x


def range_tensor(end: int) -> Tensor:
    """Create a PyTorch tensor containing values within given range"""
    return torch.arange(end).long().to(Config.DEVICE)


def to_np(tensor: Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a Numpy array"""
    return tensor.cpu().detach().numpy()


def random_seed(seed: int = None) -> None:
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(int(1e6)))


def set_one_thread() -> None:
    """Set number of threads to 1 for efficiency"""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


def huber(x: Tensor, k: float = 1.0) -> Tensor:
    """Compute Huber loss

    Huber loss = 0.5 * (x ** 2),  if |x| < k
                 k*(|x| - 0.5 * k),  otherwise
    """
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


def epsilon_greedy(epsilon: float, x: Union[Tensor, np.ndarray]) -> int:
    """Selects action randomly with probability epsilon, otherwise selects greedy action"""
    if len(x.shape) == 1:
        return np.random.randint(len(x)) if np.random.rand() < epsilon else np.argmax(x)
    elif len(x.shape) == 2:
        random_actions = np.random.randint(x.shape[1], size=x.shape[0])
        greedy_actions = np.argmax(x, axis=-1)
        dice = np.random.rand(x.shape[0])
        return np.where(dice < epsilon, random_actions, greedy_actions)


def sync_grad(target_network: torch.nn.Module, src_network: torch.nn.Module) -> None:
    """Copy gradients from source network to target network"""
    for param, src_param in zip(target_network.parameters(), src_network.parameters()):
        if src_param.grad is not None:
            param._grad = src_param.grad.clone()


def batch_diagonal(input: Tensor) -> Tensor:
    """Create batch matrix of stacked diagonal matrices from a batch of vectors"""
    sizes = input.size()
    sizes = sizes + sizes[-1:]
    output = torch.zeros(sizes, device=input.device)
    strides = [output.stride(i) for i in range(input.dim() - 1)]
    strides.append(output.size(-1) + 1)
    output.as_strided(input.size(), strides).copy_(input)
    return output


def batch_trace(input: Tensor) -> Tensor:
    """Compute trace of a batch of matrices"""
    indices = range_tensor(input.size(-1))
    trace = input[:, indices, indices].sum(-1).unsqueeze(-1).unsqueeze(-1)
    return trace


class DiagonalNormal:
    """A wrapper around PyTorch's Normal distribution"""

    def __init__(self, mean: Tensor, std: Tensor):
        self.dist = Normal(mean, std)
        self.sample = self.dist.sample

    def log_prob(self, action: Tensor) -> Tensor:
        """Return log of probability density/mass function"""
        return self.dist.log_prob(action).sum(-1).unsqueeze(-1)

    def entropy(self) -> Tensor:
        """Return entropy of distribution"""
        return self.dist.entropy().sum(-1).unsqueeze(-1)

    def cdf(self, action: Tensor) -> Tensor:
        """Return cumulative distribution function"""
        return self.dist.cdf(action).prod(-1).unsqueeze(-1)


class BatchCategorical:
    """Wrapper around PyTorch's Categorical distribution"""

    def __init__(self, logits: Tensor):
        self.pre_shape = logits.size()[:-1]
        logits = logits.view(-1, logits.size(-1))
        self.dist = Categorical(logits=logits)

    def log_prob(self, action: Tensor) -> Tensor:
        """Return log of probability density/mass function"""
        log_prob = self.dist.log_prob(action.view(-1))
        log_prob = log_prob.view(action.size()[:-1] + (-1,))
        return log_prob

    def entropy(self) -> Tensor:
        """Return entropy of distribution"""
        entropy = self.dist.entropy()
        entropy = entropy.view(self.pre_shape + (-1,))
        return entropy

    def sample(self, sample_shape: torch.Size = torch.Size([])) -> Tensor:
        """Sample from the distribution"""
        sample = self.dist.sample(sample_shape)
        sample = sample.view(sample_shape + self.pre_shape + (-1,))
        return sample


class Grad:
    """Class to handle gradients"""

    def __init__(self, network: torch.nn.Module = None, grads=None):
        if grads is not None:
            self.grads = grads
        else:
            self.grads = []
            for param in network.parameters():
                self.grads.append(torch.zeros(param.data.size(), device=Config.DEVICE))

    def add(self, op):
        """Add gradients"""
        if isinstance(op, Grad):
            for grad, op_grad in zip(self.grads, op.grads):
                grad.add_(op_grad)
        elif isinstance(op, torch.nn.Module):
            for grad, param in zip(self.grads, op.parameters()):
                if param.grad is not None:
                    grad.add_(param.grad)
        return self

    def mul(self, coef: float) -> None:
        """Multiply gradients by a scalar coefficient"""
        for grad in self.grads:
            grad.mul_(coef)

    def assign(self, network: torch.nn.Module) -> None:
        """Assign gradients to the network parameters"""
        for grad, param in zip(self.grads, network.parameters()):
            param._grad = grad.clone()

    def zero(self) -> None:
        """Reset gradients to zero"""
        for grad in self.grads:
            grad.zero_()

    def clone(self):
        """Return a deep copy of the instance"""
        return Grad(grads=[grad.clone() for grad in self.grads])


class Grads:
    """Class to handle multiple gradients"""
    
    def __init__(self, network: torch.nn.Module = None, n: int = 0, grads=None):
        if grads is not None:
            self.grads = grads
        else:
            self.grads = [Grad(network) for _ in range(n)]

    def clone(self):
        """Return a deep copy of the instance"""
        return Grads(grads=[grad.clone() for grad in self.grads])

    def mul(self, op):
        """Multiply by scalar/tensor"""
        if np.isscalar(op):
            for grad in self.grads:
                grad.mul(op)
        elif isinstance(op, Tensor):
            op = op.view(-1)
            for i, grad in enumerate(self.grads):
                grad.mul(op[i])
        else:
            raise NotImplementedError
        return self

    def add(self, op):
        """Add scalar/tensor/Grads object"""
        if np.isscalar(op):
            for grad in self.grads:
                grad.mul(op)
        elif isinstance(op, Grads):
            for grad, op_grad in zip(self.grads, op.grads):
                grad.add(op_grad)
        elif isinstance(op, Tensor):
            op = op.view(-1)
            for i, grad in enumerate(self.grads):
                grad.mul(op[i])
        else:
            raise NotImplementedError
        return self

    def mean(self) -> Grad:
        """Compute the mean of the gradients"""
        grad = self.grads[0].clone()
        grad.zero()
        for g in self.grads:
            grad.add(g)
        grad.mul(1 / len(self.grads))
        return grad


def escape_float(x: float) -> str:
    """Replace decimal point with backslash and decimal"""
    return ('%s' % x).replace('.', '\.') 


if __name__ == "__main__":
    pass