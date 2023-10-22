from .config import Config
import numpy as np
import torch
import os


def select_device(gpu_id):
    """
    Sets the device for torch tensors based on the GPU available.

    Args:
        gpu_id (int): GPU id to use. -1 for CPU.

    """
    if gpu_id >= 0:
        Config.DEVICE = torch.device('cuda:%d' % gpu_id)
    else:
        Config.DEVICE = torch.device('cpu')


def to_tensor(x):
    """
    Converts a numpy array to a tensor on the configured device.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        torch.Tensor: Tensor on the configured device.

    """
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).to(Config.DEVICE)
    return x


def range_tensor(end):
    """
    Returns a tensor with values in the range [0, end).

    Args:
        end (int): End value (exclusive).

    Returns:
        torch.Tensor: Tensor of integer values in the range [0, end).

    """
    return torch.arange(end).long().to(Config.DEVICE)


def to_numpy(t):
    """
    Converts a tensor to a numpy array.

    Args:
        t (torch.Tensor): Input tensor.

    Returns:
        numpy.ndarray: Numpy array.

    """
    return t.cpu().detach().numpy()


def set_random_seed(seed=None):
    """
    Sets the random seed for numpy and torch.

    Args:
        seed (int, optional): Seed value. Defaults to None.

    """
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(int(1e6)))


def set_one_thread():
    """
    Sets the environment variables to use one thread.

    """
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


def huber_loss(x, k=1.0):
    """
    Computes the Huber loss.

    Args:
        x (torch.Tensor): Input tensor.
        k (float, optional): Threshold value. Defaults to 1.0.

    Returns:
        torch.Tensor: Huber loss tensor.

    """
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


def epsilon_greedy(epsilon, x):
    """
    Implements epsilon-greedy exploration strategy.

    Args:
        epsilon (float): Probability of selecting a random action.
        x (numpy.ndarray): Action values.

    Returns:
        int or numpy.ndarray: Selected action(s).

    """
    if len(x.shape) == 1:
        return np.random.randint(len(x)) if np.random.rand() < epsilon else np.argmax(x)
    elif len(x.shape) == 2:
        random_actions = np.random.randint(x.shape[1], size=x.shape[0])
        greedy_actions = np.argmax(x, axis=-1)
        dice = np.random.rand(x.shape[0])
        return np.where(dice < epsilon, random_actions, greedy_actions)


def sync_gradients(target_network, src_network):
    """
    Synchronizes the gradients of two neural networks.

    Args:
        target_network (torch.nn.Module): Target network.
        src_network (torch.nn.Module): Source network.

    """
    for param, src_param in zip(target_network.parameters(), src_network.parameters()):
        if src_param.grad is not None:
            param.grad = src_param.grad.clone()


def batch_diagonal(input):
    """
    Batch diagonal function.

    Args:
        input (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Diagonal tensor.

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
    Computes the trace of a batch of square matrices.

    Args:
        input (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Trace tensor.

    """
    i = range_tensor(input.size(-1))
    t = input[:, i, i].sum(-1).unsqueeze(-1).unsqueeze(-1)
    return t


class DiagonalNormal:
    """
    Diagonal normal distribution.

    """
    def __init__(self, mean, std):
        """
        Initializes the diagonal normal distribution.

        Args:
            mean (torch.Tensor): Mean tensor.
            std (torch.Tensor): Standard deviation tensor.

        """
        self.dist = torch.distributions.Normal(mean, std)
        self.sample = self.dist.sample

    def log_prob(self, action):
        """
        Computes the log probability of an action.

        Args:
            action (torch.Tensor): Action tensor.

        Returns:
            torch.Tensor: Log probability tensor.

        """
        return self.dist.log_prob(action).sum(-1).unsqueeze(-1)

    def entropy(self):
        """
        Computes the entropy of the distribution.

        Returns:
            torch.Tensor: Entropy tensor.

        """
        return self.dist.entropy().sum(-1).unsqueeze(-1)

    def cdf(self, action):
        """
        Computes the cumulative distribution function of the distribution.

        Args:
            action (torch.Tensor): Action tensor.

        Returns:
            torch.Tensor: CDF tensor.

        """
        return self.dist.cdf(action).prod(-1).unsqueeze(-1)


class BatchCategorical:
    """
    Categorical distribution for a batch of samples.

    """
    def __init__(self, logits):
        """
        Initializes the categorical distribution.

        Args:
            logits (torch.Tensor): Logits tensor.

        """
        self.pre_shape = logits.size()[:-1]
        logits = logits.view(-1, logits.size(-1))
        self.dist = torch.distributions.Categorical(logits=logits)

    def log_prob(self, action):
        """
        Computes the log probability of an action.

        Args:
            action (torch.Tensor): Action tensor.

        Returns:
            torch.Tensor: Log probability tensor.

        """
        log_pi = self.dist.log_prob(action.view(-1))
        log_pi = log_pi.view(action.size()[:-1] + (-1,))
        return log_pi

    def entropy(self):
        """
        Computes the entropy of the distribution.

        Returns:
            torch.Tensor: Entropy tensor.

        """
        ent = self.dist.entropy()
        ent = ent.view(self.pre_shape + (-1,))
        return ent

    def sample(self, sample_shape=torch.Size([])):
        """
        Samples from the distribution.

        Args:
            sample_shape (torch.Size, optional): Shape of the samples. Defaults to torch.Size([]).

        Returns:
            torch.Tensor: Sample tensor.

        """
        ret = self.dist.sample(sample_shape)
        ret = ret.view(sample_shape + self.pre_shape + (-1,))
        return ret


class Grad:
    """
    Gradient class.

    """
    def __init__(self, network=None, grads=None):
        """
        Initializes the gradients.

        Args:
            network (torch.nn.Module, optional): Neural network. Defaults to None.
            grads (list of torch.Tensor, optional): List of gradient tensors. Defaults to None.

        """
        if grads is not None:
            self.grads = grads
        else:
            self.grads = []
            for param in network.parameters():
                self.grads.append(torch.zeros(param.data.size(), device=Config.DEVICE))

    def add(self, op):
        """
        Adds gradients.

        Args:
            op (Grad or torch.nn.Module): Object to add gradients from.

        Returns:
            Grad: Self object.

        """
        if isinstance(op, Grad):
            for grad, op_grad in zip(self.grads, op.grads):
                grad.add_(op_grad)
        elif isinstance(op, torch.nn.Module):
            for grad, param in zip(self.grads, op.parameters()):
                if param.grad is not None:
                    grad.add_(param.grad)
        return self

    def mul(self, coef):
        """
        Multiplies gradients with a coefficient.

        Args:
            coef (float): Multiplication coefficient.

        Returns:
            Grad: Self object.

        """
        for grad in self.grads:
            grad.mul_(coef)
        return self

    def assign(self, network):
        """
        Assigns the gradients to the network.

        Args:
            network (torch.nn.Module): Neural network.

        """
        for grad, param in zip(self.grads, network.parameters()):
            param.grad = grad.clone()

    def zero(self):
        """
        Zeros out the gradient tensor.

        """
        for grad in self.grads:
            grad.zero_()

    def clone(self):
        """
        Clones the gradient object.

        Returns:
            Grad: Cloned object.

        """
        return Grad(grads=[grad.clone() for grad in self.grads])


class Grads:
    """
    Gradients class.

    """
    def __init__(self, network=None, n=0, grads=None):
        """
        Initializes the Grads object.

        Args:
            network (torch.nn.Module, optional): Neural network. Defaults to None.
            n (int, optional): Number of gradients to store. Defaults to 0.
            grads (list of Grad, optional): List of gradients. Defaults to None.

        """
        if grads is not None:
            self.grads = grads
        else:
            self.grads = [Grad(network) for _ in range(n)]

    def clone(self):
        """
        Clones the Grads object.

        Returns:
            Grads: Cloned object.

        """
        return Grads(grads=[grad.clone() for grad in self.grads])

    def mul(self, op):
        """
        Multiplies the gradients with a coefficient.

        Args:
            op (float or torch.Tensor): Coefficient to multiply with.

        Returns:
            Grads: Self object.

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

    def add(self, op):
        """
        Adds gradients.

        Args:
            op (float, Grads or torch.Tensor): Object to add gradients from.

        Returns:
            Grads: Self object.

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

    def mean(self):
        """
        Computes the mean gradient.

        Returns:
            Grad: Mean gradient object.

        """
        grad = self.grads[0].clone()
        grad.zero()
        for g in self.grads:
            grad.add(g)
        grad.mul(1 / len(self.grads))
        return grad


def escape_float(x):
    """
    Escapes float values in string representation.

    Args:
        x (float): Input float value.

    Returns:
        str: String representation with escaped float.

    """
    return str(x).replace('.', '\.')