from typing import Union, Tuple
import torch
import numpy as np
from .config import Config


def select_device(gpu_id: int) -> None:
    '''
    Selects the specified GPU or CPU as the device for computations.
    
    Args:
    - gpu_id (int): The ID of the GPU to use for computations.
    
    Returns:
    - None
    '''
    if gpu_id >= 0:
        Config.DEVICE = torch.device(f'cuda:{gpu_id}')
    else:
        Config.DEVICE = torch.device('cpu')


def tensor(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    '''
    Converts a numpy array or pytorch tensor to a pytorch tensor on the selected device.
    
    Args:
    - x (Union[np.ndarray, torch.Tensor]): The input array or tensor.
    
    Returns:
    - (torch.Tensor): The pytorch tensor on the selected device.
    '''
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).to(Config.DEVICE)
    return x


def range_tensor(end: int) -> torch.Tensor:
    '''
    Returns a tensor containing integers from 0 to (end - 1).
    
    Args:
    - end (int): The end value of the range.
    
    Returns:
    - (torch.Tensor): A tensor containing integers from 0 to (end - 1).
    '''
    return torch.arange(end).long().to(Config.DEVICE)


def to_np(t: torch.Tensor) -> np.ndarray:
    '''
    Converts a pytorch tensor to a numpy array.
    
    Args:
    - t (torch.Tensor): The input pytorch tensor.
    
    Returns:
    - (np.ndarray): The converted numpy array.
    '''
    return t.cpu().detach().numpy()


def random_seed(seed: int = None) -> None:
    '''
    Sets the random seed for numpy and pytorch.
    
    Args:
    - seed (int): The random seed value. If None, uses a random integer between 0 and 1e6.
    
    Returns:
    - None
    '''
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(int(1e6)))


def set_one_thread() -> None:
    '''
    Sets the number of OpenMP and Intel MKL threads to 1 for multi-threading optimization.
    
    Args:
    - None
    
    Returns:
    - None
    '''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


def huber(x: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    '''
    Computes the Huber loss for the input tensor.
    
    Args:
    - x (torch.Tensor): The input tensor.
    - k (float): The Huber loss threshold value.
    
    Returns:
    - (torch.Tensor): The computed Huber loss tensor.
    '''
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


def epsilon_greedy(epsilon: float, x: Union[np.ndarray, torch.Tensor]) -> int:
    '''
    Implements the epsilon-greedy exploration strategy for selecting actions in a multi-armed bandit problem.
    
    Args:
    - epsilon (float): The exploration probability epsilon (0 <= epsilon <= 1).
    - x (Union[np.ndarray, torch.Tensor]): The Q-values for the available actions.
    
    Returns:
    - (int): The selected action index.
    '''
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(Config.DEVICE)
    if len(x.shape) == 1:
        return np.random.randint(len(x)) if np.random.rand() < epsilon else torch.argmax(x).item()
    elif len(x.shape) == 2:
        random_actions = np.random.randint(x.shape[1], size=x.shape[0])
        greedy_actions = torch.argmax(x, dim=-1).cpu().detach().numpy()
        dice = np.random.rand(x.shape[0])
        return np.where(dice < epsilon, random_actions, greedy_actions)


def sync_grad(target_model: torch.nn.Module, src_model: torch.nn.Module) -> None:
    '''
    Synchronizes the gradients of the target model with those of the source model.
    
    Args:
    - target_model (torch.nn.Module): The target model to update.
    - src_model (torch.nn.Module): The source model to copy the gradients from.
    
    Returns:
    - None
    '''
    for param, src_param in zip(target_model.parameters(), src_model.parameters()):
        if src_param.grad is not None:
            param.grad = src_param.grad.clone()


def batch_diagonal(input: torch.Tensor) -> torch.Tensor:
    '''
    Computes a batch of diagonal matrices from a batch of vectors.
    
    Args:
    - input (torch.Tensor): A tensor with shape (batch_size, d).
    
    Returns:
    - (torch.Tensor): A tensor with shape (batch_size, d, d) representing batch of diagonal matrices.
    '''
    # idea from here: https://discuss.pytorch.org/t/batch-of-diagonal-matrix/13560
    # batches a stack of vectors (batch x N) -> a stack of diagonal matrices (batch x N x N)
    # works in  2D -> 3D, should also work in higher dimensions
    # make a zero matrix, which duplicates the last dim of input
    dims = input.size()
    dims = dims + dims[-1:]
    output = torch.zeros(dims, device=input.device)
    # stride across the first dimensions, add one to get the diagonal of the last dimension
    strides = [output.stride(i) for i in range(input.dim() - 1)]
    strides.append(output.size(-1) + 1)
    # stride and copy the input to the diagonal
    output.as_strided(input.size(), strides).copy_(input)
    return output


def batch_trace(input: torch.Tensor) -> torch.Tensor:
    '''
    Computes the trace of a batch of square matrices.
    
    Args:
    - input (torch.Tensor): A tensor with shape (batch_size, d, d).
    
    Returns:
    - (torch.Tensor): A tensor with shape (batch_size, 1, 1) representing the trace of each matrix in the batch.
    '''
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
        action = action.view(-1)
        log_pi = self.dist.log_prob(action)
        log_pi = log_pi.view(*self.pre_shape, -1)
        return log_pi

    def entropy(self) -> torch.Tensor:
        ent = self.dist.entropy()
        ent = ent.view(*self.pre_shape, -1)
        return ent

    def sample(self, sample_shape: Tuple = ()) -> torch.Tensor:
        ret = self.dist.sample(sample_shape)
        ret = ret.view(sample_shape + self.pre_shape + (-1,))
        return ret


class Grad:
    def __init__(self, network: torch.nn.Module, grads: Union[None, List[torch.Tensor]] = None):
        if grads is not None:
            self.grads = grads
        else:
            self.grads = []
            for param in network.parameters():
                self.grads.append(torch.zeros(param.data.size(), device=Config.DEVICE))

    def add(self, op: Union['Grad', torch.nn.Module]) -> 'Grad':
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
            param.grad = grad.clone()

    def zero(self) -> None:
        for grad in self.grads:
            grad.zero_()

    def clone(self) -> 'Grad':
        return Grad(grads=[grad.clone() for grad in self.grads])


class Grads:
    def __init__(self, network: torch.nn.Module, n: int = 0, grads: Union[None, List[Grad]] = None):
        if grads is not None:
            self.grads = grads
        else:
            self.grads = [Grad(network) for _ in range(n)]

    def clone(self) -> 'Grads':
        return Grads(grads=[grad.clone() for grad in self.grads])

    def mul(self, op: Union[float, torch.Tensor]) -> 'Grads':
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

    def add(self, op: Union[float, 'Grads', torch.Tensor]) -> 'Grads':
        if np.isscalar(op):
            for grad in self.grads:
                grad.add(op)
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
        grad.zero()
        for g in self.grads:
            grad.add(g)
        grad.mul(1 / len(self.grads))
        return grad


def escape_float(x: Union[float, torch.Tensor]) -> str:
