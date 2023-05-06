import torch
import numpy as np


def set_device(gpu_id):
    # Sets the device to CPU or GPU depending on the availability and input.
    if gpu_id >= 0:
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    return device


def tensor(x, device):
    # Converts the input to torch tensor and transfers it to the specified device.
    if isinstance(x, torch.Tensor):
        return x.to(device)
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).to(device)
    return x


def range_tensor(end, device):
    # Creates a range tensor and transfers it to the specified device.
    return torch.arange(end).long().to(device)


def to_numpy_array(tensor):
    # Converts a tensor to a numpy array.
    return tensor.cpu().detach().numpy()


def set_seed(seed=None):
    # Sets the seed for the random number generators.
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(int(1e6)))


def disable_parallelization():
    # Disables parallelism to prevent non-deterministic behavior.
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


def huber_loss(x, k=1.0):
    # Computes the Huber loss.
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


def epsilon_greedy(epsilon, x):
    # Performs epsilon-greedy exploration for the input action values.
    if len(x.shape) == 1:
        return np.random.randint(len(x)) if np.random.rand() < epsilon else np.argmax(x)
    elif len(x.shape) == 2:
        random_actions = np.random.randint(x.shape[1], size=x.shape[0])
        greedy_actions = np.argmax(x, axis=-1)
        dice = np.random.rand(x.shape[0])
        return np.where(dice < epsilon, random_actions, greedy_actions)


def sync_grads(target_network, src_network):
    # Synchronizes the gradients of the target network with the source network.
    for target_param, src_param in zip(target_network.parameters(), src_network.parameters()):
        if src_param.grad is not None:
            target_param.grad = src_param.grad.clone()


def batch_diag(input):
    # Batches a stack of vectors (batch x N) -> a stack of diagonal matrices (batch x N x N).
    # Works in  2D -> 3D, should also work in higher dimensions.
    # Adapted from https://discuss.pytorch.org/t/batch-of-diagonal-matrix/13560
    batch_dims = input.size()[:-1]
    output_dims = batch_dims + (input.shape[-1],)
    output = torch.zeros(output_dims, device=input.device)
    strides = [output.stride(i) for i in range(input.dim() - 1)]
    strides.append(output.size(-1) + 1)
    output.as_strided(input.size(), strides).copy_(input)
    return output


def batch_trace(input):
    # Computes the trace of each square matrix in a batch of square matrices.
    # Assumes that input has shape (batch_size, M, M).
    i = range_tensor(input.size(-1), input.device)
    t = input[:, i, i].sum(-1).unsqueeze(-1).unsqueeze(-1)
    return t


class DiagonalNormal:
    # Wrapper class for the diagonal normal distribution for Torch tensors.
    def __init__(self, mean, std):
        self.dist = torch.distributions.Normal(mean, std)
        self.sample = self.dist.sample

    def log_prob(self, action):
        return self.dist.log_prob(action).sum(-1).unsqueeze(-1)

    def entropy(self):
        return self.dist.entropy().sum(-1).unsqueeze(-1)

    def cdf(self, action):
        return self.dist.cdf(action).prod(-1).unsqueeze(-1)


class BatchCategorical:
    # Wrapper class for the batch categorial distribution for Torch tensors.
    def __init__(self, logits):
        self.pre_shape = logits.size()[:-1]
        logits = logits.view(-1, logits.size(-1))
        self.dist = torch.distributions.Categorical(logits=logits)

    def log_prob(self, action):
        log_pi = self.dist.log_prob(action.view(-1))
        log_pi = log_pi.view(action.size()[:-1] + (-1,))
        return log_pi

    def entropy(self):
        ent = self.dist.entropy()
        ent = ent.view(self.pre_shape + (-1,))
        return ent

    def sample(self, sample_shape=torch.Size([])):
        ret = self.dist.sample(sample_shape)
        ret = ret.view(sample_shape + self.pre_shape + (-1,))
        return ret


class Grad:
    # Container class for gradients of a single network.
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
    # Container class for multiple gradients of a network.
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
        grad = self.grads[0].clone()
        grad.zero()
        for g in self.grads:
            grad.add(g)
        grad.mul_(1 / len(self.grads))
        return grad


def escape_float(x):
    # Escapes floats in a string.
    return ('%s' % x).replace('.', '\.')