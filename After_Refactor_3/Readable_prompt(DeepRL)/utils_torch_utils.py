# Improved and refactored code

# file: utils.py

import numpy as np
import torch
import os
from .config import *


def select_device(gpu_id: int) -> None:
    """
    Given a gpu id, it assigns the torch device.
    """
    if gpu_id >= 0:
        Config.DEVICE = torch.device(f'cuda:{gpu_id}')
    else:
        Config.DEVICE = torch.device('cpu')


def tensor(x):
    """
    Data type conversion to torch tensor
    """
    x = np.asarray(x, dtype=np.float32)
    if isinstance(x, torch.Tensor):
        return x
    x = torch.from_numpy(x).to(Config.DEVICE)
    return x


def to_np(t: torch.Tensor) -> np.ndarray:
    """
    torch tensor to numpy data type conversion
    """
    return t.cpu().detach().numpy()


def random_seed(seed=None):
    """
    Seeds the random number generator used by numpy and Torch.
    """
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(int(1e6)))


def huber(x: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """
    Returns the Huber loss.
    """
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


def epsilon_greedy(epsilon, x):
    """
    epsilon-greedy strategy
    """
    if len(x.shape) == 1:
        return np.random.randint(len(x)) if np.random.rand() < epsilon else np.argmax(x)
    elif len(x.shape) == 2:
        random_actions = np.random.randint(x.shape[1], size=x.shape[0])
        greedy_actions = np.argmax(x, axis=-1)
        dice = np.random.rand(x.shape[0])
        return np.where(dice < epsilon, random_actions, greedy_actions)


def sync_grad(target_network, src_network) -> None:
    """
    Copy over the gradients of the source network to the target.
    """
    for param, src_param in zip(target_network.parameters(), src_network.parameters()):
        if src_param.grad is not None:
            param._grad = src_param.grad.clone()


def escape_float(x: str) -> str:
    """
    Returns a string and replaces '.' with '\.'.
    """
    return ('%s' % x).replace('.', '\.')


def batch_diagonal(input: torch.Tensor) -> torch.Tensor:
    """
    It creates a diagonal matrix from a vector.
    """
    dims = input.size()
    dims = dims + dims[-1:]
    output = torch.zeros(dims, device=input.device)
    strides = [output.stride(i) for i in range(input.dim() - 1)]
    strides.append(output.size(-1) + 1)
    output.as_strided(input.size(), strides).copy_(input)
    return output


def batch_trace(input: torch.Tensor) -> torch.Tensor:
    """
    Returns the trace of a matrix.
    """
    i = range_tensor(input.size(-1))
    t = input[:, i, i].sum(-1).unsqueeze(-1).unsqueeze(-1)
    return t


class DiagonalNormal:
    
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.dist = torch.distributions.Normal(mean, std)
        self.sample = self.dist.sample
    
    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(action).sum(-1).unsqueeze(-1)
    
    def entropy(self) -> torch.Tensor:
        return self.dist.entropy().sum(-1).unsqueeze(-1)
    
    def cdf(self, action: torch.Tensor) -> torch.Tensor:
        return self.dist.cdf(action).prod(-1).unsqueeze(-1)


class BatchCategorical:
    
    def __init__(self, logits: torch.Tensor) -> None:
        self.pre_shape = logits.size()[:-1]
        logits = logits.view(-1, logits.size(-1))
        self.dist = torch.distributions.Categorical(logits=logits)
        
    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        log_pi = self.dist.log_prob(action.view(-1))
        log_pi = log_pi.view(action.size()[:-1] + (-1,))
        return log_pi
    
    def entropy(self) -> torch.Tensor:
        ent = self.dist.entropy()
        ent = ent.view(self.pre_shape + (-1,))
        return ent
    
    def sample(self, sample_shape=torch.Size([])) -> torch.Tensor:
        ret = self.dist.sample(sample_shape)
        ret = ret.view(sample_shape + self.pre_shape + (-1,))
        return ret


class Grad:
    """
    Gradient calculations.
    """
    def __init__(self, network=None, grads=None):
        if grads is not None:
            self.grads = grads
        else:
            self.grads = []
            for param in network.parameters():
                self.grads.append(torch.zeros(param.data.size(), device=Config.DEVICE))

    def add(self, op) -> 'Grad':
        """
        Adds two different gradients.
        """
        if isinstance(op, Grad):
            for grad, op_grad in zip(self.grads, op.grads):
                grad.add_(op_grad)
        elif isinstance(op, torch.nn.Module):
            for grad, param in zip(self.grads, op.parameters()):
                if param.grad is not None:
                    grad.add_(param.grad)
        return self

    def mul(self, coef: float) -> 'Grad':
        """
        Multiplies a gradient by a scalar.
        """
        for grad in self.grads:
            grad.mul_(coef)
        return self

    def assign(self, network) -> None:
        """
        Assigns the calculated gradient to the model's parameters
        """
        for grad, param in zip(self.grads, network.parameters()):
            param._grad = grad.clone()

    def zero(self) -> None:
        """
        Resets all gradients in the instance to zero.
        """
        for grad in self.grads:
            grad.zero_()

    def clone(self) -> 'Grad':
        """
        returns a copy of the class instance.
        """
        return Grad(grads=[grad.clone() for grad in self.grads])


class Grads:
    """
    Values of the gradient in each epoch (or mini-batch)
    """
    def __init__(self, network=None, n=0, grads=None):
        if grads is not None:
            self.grads = grads
        else:
            self.grads = [Grad(network) for _ in range(n)]

    def clone(self) -> 'Grads':
        """
        returns a copy of the class instance.
        """
        return Grads(grads=[grad.clone() for grad in self.grads])

    def mul(self, op) -> 'Grads':
        """
        Multiplies the grads by a scalar or a tensor.
        """
        if np.isscalar(op):
            for grad in self.grads:
                grad.mul(op)
        elif isinstance(op, torch.Tensor):
            op = op.view(-1)
            for i, grad in enumerate(self.grads):
                grad.mul(op[i])
        else:
            raise NotImplementedError
        return self

    def add(self, op) -> 'Grads':
        """
        Adds two grads of the same instance of the class.
        """
        if np.isscalar(op):
            for grad in self.grads:
                grad.mul(op)
        elif isinstance(op, Grads):
            for grad, op_grad in zip(self.grads, op.grads):
                grad.add(op_grad)
        elif isinstance(op, torch.Tensor):
            op = op.view(-1)
            for i, grad in enumerate(self.grads):
                grad.mul(op[i])
        else:
            raise NotImplementedError
        return self

    def mean(self) -> Grad:
        """
        Calculates the average of the grads.
        """
        grad = self.grads[0].clone()
        grad.zero()
        for g in self.grads:
            grad.add(g)
        grad.mul(1 / len(self.grads))
        return grad
