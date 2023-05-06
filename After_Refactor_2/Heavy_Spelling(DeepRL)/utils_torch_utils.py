import os
import numpy as np
import torch
from .config import Config


def select_device(gpu_id):
    """
    Selects the device to use based on the GPU ID passed as argument.
    """
    if gpu_id >= 0:
        Config.DEVICE = torch.device(f"cuda:{gpu_id}")
    else:
        Config.DEVICE = torch.device("cpu")


def tensor(x):
    """
    Converts the input to a PyTorch tensor.
    """
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).to(Config.DEVICE)
    return x


def range_tensor(end):
    """
    Returns a tensor containing a sequence of integers from 0 to end.
    """
    return torch.arange(end).long().to(Config.DEVICE)


def to_np(tensor):
    """
    Converts the PyTorch tensor to a numpy array.
    """
    return tensor.cpu().detach().numpy()


def random_seed(seed=None):
    """
    Sets the seed for the random number generators used in the code.
    """
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(int(1e6)))


def set_one_thread():
    """
    Sets the number of OpenMP and MKL threads to 1 to improve performance.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def huber_loss(x, k=1.0):
    """
    Calculates the Huber loss for the input tensor x.
    """
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


def epsilon_greedy(epsilon, x):
    """
    Returns an action index based on an epsilon-greedy policy.
    """
    if len(x.shape) == 1:
        return np.random.randint(len(x)) if np.random.rand() < epsilon else np.argmax(x)
    elif len(x.shape) == 2:
        random_actions = np.random.randint(x.shape[1], size=x.shape[0])
        greedy_actions = np.argmax(x, axis=-1)
        dice = np.random.rand(x.shape[0])
        return np.where(dice < epsilon, random_actions, greedy_actions)


def sync_grad(target_network, src_network):
    """
    Copies the gradients from the source network to the target network.
    """
    for param, src_param in zip(target_network.parameters(), src_network.parameters()):
        if src_param.grad is not None:
            param.grad = src_param.grad.clone()


def batch_diagonal(input):
    """
    Converts a batch of vectors to a batch of diagonal matrices.
    """
    dims = input.size()
    dims = dims + dims[-1:]
    output = torch.zeros(dims, device=input.device)
    strides = [output.stride(i) for i in range(input.dim() - 1)]
    strides.append(output.size(-1) + 1)
    output.as_strided(input.size(), strides).copy_(input)
    return output


def batch_trace(input):
    """
    Calculates the trace of a batch of square matrices.
    """
    i = range_tensor(input.size(-1))
    trace = input[:, i, i].sum(-1).unsqueeze(-1).unsqueeze(-1)
    return trace


class DiagonalNormal:
    """
    A diagonal Gaussian distribution.
    """
    def __init__(self, mean, std):
        self.dist = torch.distributions.Normal(mean, std)

    def log_prob(self, action):
        return self.dist.log_prob(action).sum(-1).unsqueeze(-1)

    def entropy(self):
        return self.dist.entropy().sum(-1).unsqueeze(-1)

    def cdf(self, action):
        return self.dist.cdf(action).prod(-1).unsqueeze(-1)


class BatchCategorical:
    """
    A batch of categorical distributions.
    """
    def __init__(self, logits):
        self.pre_shape = logits.size()[:-1]
        logits = logits.view(-1, logits.size(-1))
        self.dist = torch.distributions.Categorical(logits=logits)

    def log_prob(self, action):
        log_prob = self.dist.log_prob(action.view(-1))
        log_prob = log_prob.view(action.size()[:-1] + (-1,))
        return log_prob

    def entropy(self):
        entropy = self.dist.entropy()
        entropy = entropy.view(self.pre_shape + (-1,))
        return entropy

    def sample(self, sample_shape=torch.Size([])):
        sample = self.dist.sample(sample_shape)
        sample = sample.view(sample_shape + self.pre_shape + (-1,))
        return sample


class Grad:
    """
    A gradient accumulator.
    """
    def __init__(self, network=None, grads=None):
        if grads is not None:
            self.grads = grads
        else:
            self.grads = [torch.zeros(param.data.size(), device=Config.DEVICE) for param in network.parameters()]

    def add(self, op):
        if isinstance(op, Grad):
            for grad, op_grad in zip(self.grads, op.grads):
                grad.add_(op_grad)
        elif isinstance(op, torch.nn.Module):
            for grad, param in zip(self.grads, op.parameters()):
                if param.grad is not None:
                    grad.add_(param.grad)
        return self

    def mul(self, coef):
        for grad in self.grads:
            grad.mul_(coef)
        return self

    def assign(self, network):
        for grad, param in zip(self.grads, network.parameters()):
            param.grad = grad.clone()

    def zero(self):
        for grad in self.grads:
            grad.zero_()

    def clone(self):
        return Grad(grads=[grad.clone() for grad in self.grads])


class Grads:
    """
    A gradient accumulator for a batch of networks.
    """
    def __init__(self, network=None, n=0, grads=None):
        if grads is not None:
            self.grads = grads
        else:
            self.grads = [Grad(network) for _ in range(n)]

    def clone(self):
        return Grads(grads=[grad.clone() for grad in self.grads])

    def mul(self, op):
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

    def add(self, op):
        if np.isscalar(op):
            for grad in self.grads:
                grad.add(op)
        elif isinstance(op, Grads):
            for grad, op_grad in zip(self.grads, op.grads):
                grad.add(op_grad)
        elif isinstance(op, torch.Tensor):
            op = op.view(-1)
            for i, grad in enumerate(self.grads):
                grad.add(op[i])
        else:
            raise NotImplementedError
        return self

    def mean(self):
        gradient_mean = self.grads[0].clone()
        gradient_mean.zero()
        for gradient in self.grads:
            gradient_mean.add(gradient)
        gradient_mean.mul(1 / len(self.grads))
        return gradient_mean


def escape_float(x):
    """
    Escapes the decimal point in a float to prevent issues with regular expressions.
    """
    return f"{x}".replace(".", ".")