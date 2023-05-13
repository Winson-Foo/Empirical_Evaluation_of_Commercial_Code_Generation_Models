import numpy as np
import torch
import os
from typing import List, Tuple


def select_device(gpu_id: int) -> None:
    """
    Select a device based on the gpu_id passed as a parameter.

    Params:
    -------
    gpu_id: int: GPU ID to select. `-1` means no GPUs available.
    """
    if gpu_id >= 0:
        Config.DEVICE = torch.device(f'cuda:{gpu_id}')
    else:
        Config.DEVICE = torch.device('cpu')


def tensor(x: np.ndarray) -> torch.Tensor:
    """
    Convert a numpy array to a PyTorch tensor.

    Params:
    -------
    x: np.ndarray: A numpy array to convert.
    
    Returns:
    -------
    tensor: torch.Tensor: A converted PyTorch tensor.
    """
    if isinstance(x, torch.Tensor):
        return x.to(Config.DEVICE)
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).to(Config.DEVICE)
    return x


def range_tensor(end: int) -> torch.Tensor:
    """
    Generate a tensor of integers ranging from 0 to `end`.

    Params:
    -------
    end: int: The end of the range.
    
    Returns:
    -------
    tensor: torch.Tensor: A tensor of integers ranging from 0 to `end`.
    """
    return torch.arange(end).long().to(Config.DEVICE)


def to_np(t: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a Numpy array.

    Params:
    -------
    t: torch.Tensor: A PyTorch tensor to convert.
    
    Returns:
    -------
    arr: np.ndarray: A Numpy array derived from the parameter.
    """
    return t.cpu().detach().numpy()


def random_seed(seed: int = None) -> None:
    """
    Set the random seed for NumPy and PyTorch.

    Args:
    ----
    seed: int: The random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(int(1e6)))


def set_one_thread() -> None:
    """
    Set the number of threads to 1 for PyTorch and MPI.
    """
    os.environ['OMP_NUM_THREADS'] = str(Config.OMP_NUM_THREADS)
    os.environ['MKL_NUM_THREADS'] = str(Config.MKL_NUM_THREADS)
    torch.set_num_threads(Config.TORCH_NUM_THREADS)


def huber(x: torch.Tensor, k: float = Config.HUBER_K) -> torch.Tensor:
    """
    A Huber function that calculates f(x) = 0.5 * x ** 2 if |x| <= k or k*(|x|-0.5*k) otherwise.
    
    Params:
    -------
    x: torch.Tensor: The input tensor to the Huber function.
    k: float: The thresholding value to differentiate between quadratic and linear loss components.
    
    Returns:
    -------
    torch.Tensor: The Huber function result calculated for the input tensor.
    """
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


def epsilon_greedy(epsilon: float, x: np.ndarray) -> List[int]:
    """
    Perform epsilon-greedy exploration strategy to select random or greedy actions.

    Params:
    -------
    epsilon: float: The probability of selecting a random action in the range `[0, 1]`.
    x: np.ndarray: A 1D or 2D input array of action values.
    
    Returns:
    -------
    action: np.ndarray: An action index selected based on the exploration strategy.
    """
    if len(x.shape) == 1:
        return np.random.randint(len(x)) if np.random.rand() < epsilon else np.argmax(x)
    elif len(x.shape) == 2:
        random_actions = np.random.randint(x.shape[1], size=x.shape[0])
        greedy_actions = np.argmax(x, axis=-1)
        dice = np.random.rand(x.shape[0])
        return np.where(dice < epsilon, random_actions, greedy_actions)


def sync_grad(target_network: torch.nn.Module, src_network: torch.nn.Module) -> None:
    """
    Copy the gradients of source network's parameters to the destination network's parameters.

    Params:
    -------
    target_network: torch.nn.Module: The recipient network.
    src_network: torch.nn.Module: The source network.
    """
    for param, src_param in zip(target_network.parameters(), src_network.parameters()):
        if src_param.grad is not None:
            param._grad = src_param.grad.clone()


def batch_diagonal(input: torch.Tensor) -> torch.Tensor:
    """
    Batch a stack of vectors four 4D tensors to stack of diagonal matrices.

    Params:
    -------
    input: torch.Tensor: A tensor with last two dimensions as vector.
    
    Returns:
    -------
    tensor: torch.Tensor: A tensor of stacked diagonal matrices.
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
    Calculate the trace of input tensor.

    Params:
    -------
    input: torch.Tensor: A tensor of shape (batch_size, n, n)

    Returns:
    -------
    trace: torch.Tensor: A tensor of shape (batch_size, 1, 1), containing the trace of the input tensor.
    """
    i = range_tensor(input.size(-1))
    t = input[:, i, i].sum(-1).unsqueeze(-1).unsqueeze(-1)
    return t


class DiagonalNormal:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.dist = torch.distributions.Normal(mean, std)
        self.sample = self.dist.sample

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(action).sum(-1).unsqueeze(-1)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy().sum(-1).unsqueeze(-1)

    def cdf(self, action: torch.Tensor) -> torch.Tensor:
        return self.dist.cdf(action).prod(-1).unsqueeze(-1)


class BatchCategorical:
    def __init__(self, logits: torch.Tensor):
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
    def __init__(self, network=None, grads: List[torch.Tensor] = None):
        if grads is not None:
            self.grads = grads
        else:
            self.grads = [torch.zeros(param.data.size(), device=Config.DEVICE) for param in network.parameters()]

    def add(self, op) -> 'Grad':
        if isinstance(op, Grad):
            for grad, op_grad in zip(self.grads, op.grads):
                grad.add_(op_grad)
        elif isinstance(op, torch.nn.Module):
            for grad, param in zip(self.grads, op.parameters()):
                if param.grad is not None:
                    grad.add_(param.grad)
        return self

    def mul(self, coef: float) -> 'Grad':
        for grad in self.grads:
            grad.mul_(coef)
        return self

    def assign(self, network: torch.nn.Module) -> None:
        for grad, param in zip(self.grads, network.parameters()):
            param._grad = grad.clone()

    def zero(self) -> None:
        for grad in self.grads:
            grad.zero_()

    def clone(self) -> 'Grad':
        return Grad(grads=[grad.clone() for grad in self.grads])


class Grads:
    def __init__(self, network=None, n: int = 0, grads: List[Grad] = None):
        if grads is not None:
            self.grads = grads
        else:
            self.grads = [Grad(network) for _ in range(n)]

    def clone(self) -> 'Grads':
        return Grads(grads=[grad.clone() for grad in self.grads])

    def mul(self, op) -> 'Grads':
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
        grad = self.grads[0].clone()
