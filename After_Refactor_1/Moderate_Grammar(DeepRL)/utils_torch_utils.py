import torch
import os

# Constants
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def select_device(gpu_id: int) -> None:
    """Selects the device for pytorch computation"""
    global DEVICE
    if gpu_id >= 0:
        DEVICE = torch.device('cuda:%d' % (gpu_id))
    else:
        DEVICE = torch.device('cpu')


def tensor(x: torch.Tensor, dtype: torch.dtype=torch.float32) -> torch.Tensor:
    """Converts numpy array to tensor"""
    if isinstance(x, torch.Tensor):
        return x
    x = torch.as_tensor(x, dtype=dtype, device=DEVICE)
    return x


def range_tensor(end: int) -> torch.Tensor:
    """Returns a tensor with range from 0 to end"""
    return torch.arange(end, device=DEVICE)


def to_np(t: torch.Tensor) -> torch.Tensor:
    """Converts tensor to numpy array"""
    return t.detach().numpy()


def random_seed(seed: int=None) -> None:
    """Sets the random seed for reproducibility"""
    if seed is not None:
        torch.manual_seed(seed)


def set_one_thread() -> None:
    """Sets the number of threads for torch to 1"""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


def huber(x: torch.Tensor, k: float=1.0) -> torch.Tensor:
    """Huber loss"""
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


def epsilon_greedy(epsilon: float, x: torch.Tensor) -> int:
    """Returns action using epsilon-greedy strategy"""
    action_shape = x.shape[:-1]
    if len(action_shape) == 1:
        return torch.randint(len(x), (1,)).item() if torch.rand(1) < epsilon else x.argmax().item()
    elif len(action_shape) == 2:
        random_actions = torch.randint(x.shape[1], action_shape, device=DEVICE)
        greedy_actions = x.argmax(-1)
        dice = torch.rand(action_shape, device=DEVICE)
        return torch.where(dice < epsilon, random_actions, greedy_actions).squeeze().tolist()


def sync_grad(target_network: torch.nn.Module, src_network: torch.nn.Module) -> None:
    """Copies gradients from source network to target network"""
    for target_param, src_param in zip(target_network.parameters(), src_network.parameters()):
        if src_param.grad is not None:
            target_param.grad = src_param.grad.clone()


# adapted from https://github.com/pytorch/pytorch/issues/12160
def batch_diagonal(input: torch.Tensor) -> torch.Tensor:
    """Returns a batch of diagonal matrices"""
    # idea from here: https://discuss.pytorch.org/t/batch-of-diagonal-matrix/13560
    # batches a stack of vectors (batch x N) -> a stack of diagonal matrices (batch x N x N)
    # works in  2D -> 3D, should also work in higher dimensions
    # make a zero matrix, which duplicates the last dim of input
    dims = input.size()
    dims = dims + dims[-1:]
    output = torch.zeros(dims, device=DEVICE, dtype=input.dtype)
    # stride across the first dimensions, add one to get the diagonal of the last dimension
    strides = [output.stride(i) for i in range(input.dim() - 1)]
    strides.append(output.size(-1) + 1)
    # stride and copy the input to the diagonal
    output.as_strided(input.size(), strides).copy_(input)
    return output


def batch_trace(input: torch.Tensor) -> torch.Tensor:
    """Returns sum of diagonal elements of each matrix in batch"""
    i = range_tensor(input.size(-1))
    t = input[:, i, i].sum(-1).unsqueeze(-1).unsqueeze(-1)
    return t


class DiagonalNormal:
    """A diagonal gaussian distribution"""
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.dist = torch.distributions.Normal(mean, std)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(action).sum(-1).unsqueeze(-1)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy().sum(-1).unsqueeze(-1)

    def cdf(self, action: torch.Tensor) -> torch.Tensor:
        return self.dist.cdf(action).prod(-1).unsqueeze(-1)


class BatchCategorical:
    """A Categorical distribution for batches of logits"""
    def __init__(self, logits: torch.Tensor):
        # store the pre-softmax shape
        self.pre_shape = logits.size()[:-1]
        # reshape the logits for the distribution
        logits = logits.reshape(-1, logits.size(-1))
        self.dist = torch.distributions.Categorical(logits=logits)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        log_pi = self.dist.log_prob(action.reshape(-1))
        log_pi = log_pi.reshape(action.size()[:-1] + (-1,))
        return log_pi

    def entropy(self) -> torch.Tensor:
        ent = self.dist.entropy()
        ent = ent.reshape(self.pre_shape + (-1,))
        return ent

    def sample(self, sample_shape: torch.Size=torch.Size([])) -> torch.Tensor:
        ret = self.dist.sample(sample_shape)
        ret = ret.reshape(sample_shape + self.pre_shape + (-1,))
        return ret


class Grad:
    """Stores the gradients of a network"""
    def __init__(self, network: torch.nn.Module=None, grads: torch.Tensor=None):
        if grads is not None:
            self.grads = grads
        else:
            self.grads = [torch.zeros(param.data.size(), device=DEVICE) for param in network.parameters()]

    def add(self, op: 'Grad' or torch.nn.Module) -> 'Grad':
        """Adds gradients from another grad object or network"""
        if isinstance(op, Grad):
            for grad, op_grad in zip(self.grads, op.grads):
                grad.add_(op_grad)
        elif isinstance(op, torch.nn.Module):
            for grad, param in zip(self.grads, op.parameters()):
                if param.grad is not None:
                    grad.add_(param.grad)
        return self

    def mul(self, coef: float) -> 'Grad':
        """Multiplies gradients with a coefficient"""
        for grad in self.grads:
            grad.mul_(coef)
        return self

    def assign(self, network: torch.nn.Module) -> None:
        """Assigns the gradients to the parameters of a network"""
        for grad, param in zip(self.grads, network.parameters()):
            param.grad = grad.clone()

    def zero(self) -> None:
        """Zeroes the gradients"""
        for grad in self.grads:
            grad.zero_()

    def clone(self) -> 'Grad':
        """Returns a deep copy of the object"""
        return Grad(grads=[grad.clone() for grad in self.grads])


class Grads:
    """Stores the Gradient objects for a set of networks"""
    def __init__(self, network: torch.nn.Module=None, n: int=0, grads: 'Grad'=None):
        if grads is not None:
            self.grads = grads
        else:
            self.grads = [Grad(network) for _ in range(n)]

    def clone(self) -> 'Grads':
        """Returns a deep copy of the object"""
        return Grads(grads=[grad.clone() for grad in self.grads])

    def mul(self, op: float or torch.Tensor) -> 'Grads':
        """Multiplies the gradients with coefficients"""
        if isinstance(op, float):
            for grad in self.grads:
                grad.mul(op)
        elif isinstance(op, torch.Tensor):
            op = op.reshape(-1)
            for i, grad in enumerate(self.grads):
                grad.mul(op[i])
        else:
            raise NotImplementedError
        return self

    def add(self, op: float or 'Grads' or torch.Tensor) -> 'Grads':
        """Adds another grads object or coefficient to the gradients"""
        if isinstance(op, float):
            for grad in self.grads:
                grad.mul(op)
        elif isinstance(op, Grads):
            for grad, op_grad in zip(self.grads, op.grads):
                grad.add(op_grad)
        elif isinstance(op, torch.Tensor):
            op = op.reshape(-1)
            for i, grad in enumerate(self.grads):
                grad.mul(op[i])
        else:
            raise NotImplementedError
        return self

    def mean(self) -> 'Grad':
        """Computes the mean of the gradients"""
        grad = self.grads[0].clone()
        grad.zero()
        for g in self.grads:
            grad.add(g)
        grad.mul(1 / len(self.grads))
        return grad


def escape_float(x: float) -> str:
    """Escapes dots in a float"""
    return ('%s' % x).replace('.', '\.')