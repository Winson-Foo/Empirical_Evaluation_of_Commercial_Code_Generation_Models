# Refactored code

# Attention: Some of these functions were not properly documented regarding their inputs and outputs. Reconsidering them could help to provide a more reliable code and maintainable code.

# imports
from .config import Config
import torch
import os
import numpy as np


def select_device(gpu_id):
    # Selects the device
    Config.DEVICE = torch.device('cuda:%d' % (gpu_id)) if gpu_id >= 0 else torch.device('cpu')


def tensor(x):
    # Transforms x into a tensor to ensure that its dtype is float32 and it is stored in the GPU.
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(Config.DEVICE) if not isinstance(x, torch.Tensor) else x.to(Config.DEVICE)


def range_tensor(end):
    # Creates a rank-1 tensor of size end. For each element tensor[i] = i.
    return torch.arange(end).long().to(Config.DEVICE)


def to_np(t):
    # Detaches the tensor t from the GPU and returns the corresponding numpy array.
    return t.cpu().detach().numpy()


def random_seed(seed=None):
    # Fix the seed for random number generation.
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(int(1e6)))


def set_one_thread():
    # Restricts PyTorch to use a single thread of execution.
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


def huber(x, k=1.0):
    # Computes an element-wise huber loss by
    # 1. setting to zero the elements of x that are closer than k to 0;
    # 2. applying the squared L2 norm to the remaining elements of x;
    # 3. adding the constant k times the L1 norm to the result of step 2.
    return torch.where(x.abs() < k, x.pow(2) * 0.5, torch.abs(x) * k - k * k * 0.5)


def epsilon_greedy(epsilon, x):
    # Chooses an action according to epsilon-greedy policy.
    # Selects a random action with probability epsilon, and the action with the highest value otherwise.
    if len(x.shape) == 1:
        return np.random.randint(len(x)) if np.random.rand() < epsilon else np.argmax(x)
    
    # Chooses a random action for each state independently, with probability epsilon, and the optimal action otherwise.
    if len(x.shape) == 2:
        random_actions = np.random.randint(x.shape[1], size=x.shape[0])
        greedy_actions = np.argmax(x, axis=-1)
        dice = np.random.rand(x.shape[0])
        return np.where(dice < epsilon, random_actions, greedy_actions)


def sync_grad(target_network, src_network):
    # Copies the gradient information from the src_network to the target_network.
    for param, src_param in zip(target_network.parameters(), src_network.parameters()):
        if src_param.grad is not None:
            param.grad = src_param.grad.clone()


# adapted from https://github.com/pytorch/pytorch/issues/12160
def batch_diagonal(input):
    # Given a two-dimensional tensor input of size (batchsize, N), returns a tensor of size (batchsize, N, N),
    # where each element input[i, j] is only present in output[i, j, j].
    dims = input.size()
    output = torch.zeros(dims + dims[-1:], device=input.device)
    strides = [output.stride(i) for i in range(input.dim() - 1)]
    strides.append(output.size(-1) + 1)
    output.as_strided(input.size(), strides).copy_(input)
    return output


def batch_trace(input):
    # Given a tensor input of size (batchsize, N, N), returns a tensor of size (batchsize, 1, 1) with the trace of each matrix.
    i = range_tensor(input.size(-1))
    t = input[:, i, i].sum(-1).unsqueeze(-1).unsqueeze(-1)
    return t


class DiagonalNormal:
    def __init__(self, mean, std):
        # Creates a diagonal multivariate normal distribution from the given means and standard deviations.
        self.dist = torch.distributions.Normal(mean, std)

    def log_prob(self, action):
        # For a given batch of actions of size (batchsize, dim), returns the logarithm of the probability density value
        # given the multivariate normal distribution in each row of the tensor (along dimension 1).
        return self.dist.log_prob(action).sum(-1).unsqueeze(-1)

    def entropy(self):
        # Returns a tensor of size (batchsize, 1, 1) with the entropy of each diagonal multivariate normal distribution.
        return self.dist.entropy().sum(-1).unsqueeze(-1)

    def cdf(self, action):
        # Computes the CDF of the diagonal multivariate distribution at a given batch of actions. 
        # It returns a tensor with the same size as action.
        return self.dist.cdf(action.view(-1)).reshape_as(action).prod(-1).unsqueeze(-1)


class BatchCategorical:
    def __init__(self, logits):
        self.pre_shape = logits.size()[:-1]
        logits_reshaped = logits.view(-1, logits.size(-1))
        self.dist = torch.distributions.Categorical(logits=logits_reshaped)

    def log_prob(self, action):
        # For a given batch of actions of size (batchsize, 1), returns the logarithm of the probability value of each action.
        # Each row of the tensor (along dimension 1) considers a different categorical distribution from the batch.
        log_pi = self.dist.log_prob(action.view(-1))
        log_pi = log_pi.view(action.size()[:-1] + (-1,))
        return log_pi

    def entropy(self):
        # Returns a tensor of size (batchsize x action_space) with the entropy of each categorical distribution.
        ent = self.dist.entropy()
        ent = ent.view(self.pre_shape + (-1,))
        return ent

    def sample(self, sample_shape=torch.Size([])):
        # Draws samples from the given categorical distribution.
        # If sample_shape is empty, it returns the MAP assignments.
        # Otherwise, it returns a batch of samples with size sample_shape.
        ret = self.dist.sample(sample_shape).view(sample_shape + self.pre_shape + (-1,))
        return ret


class Grad:
    def __init__(self, network=None, grads=None):
        # Initializes a Grad object as an empty list of gradients either with the size of the given network if it is not None
        # or with the list of gradients grads.
        if grads is not None:
            self.grads = grads
        else:
            self.grads = []
            for param in network.parameters():
                self.grads.append(torch.zeros(param.data.size(), device=Config.DEVICE))

    def add(self, op):
        # If the input is another Grad object, it sums element-wise the corresponding gradient lists.
        # Otherwise, if the input is a PyTorch module, it sums element-wise the corresponding gradient tensors of the module with each gradient of the Grad object.
        if isinstance(op, Grad):
            for grad, op_grad in zip(self.grads, op.grads):
                grad.add_(op_grad)
        elif isinstance(op, torch.nn.Module):
            for grad, param in zip(self.grads, op.parameters()):
                if param.grad is not None:
                    grad.add_(param.grad)
        return self

    def mul(self, coef):
        # Multiplies each gradient tensor by the given scalar coef.
        for grad in self.grads:
            grad.mul_(coef)
        return self

    def assign(self, network):
        # Assigns the gradients stored in the Grad object to the corresponding parameters of the network.
        for grad, param in zip(self.grads, network.parameters()):
            param.grad = grad.clone()

    def zero(self):
        # Sets each gradient tensor in the Grad object to zero.
        for grad in self.grads:
            grad.zero_()

    def clone(self):
        # Creates and returns a new Grad object with copies of all the gradients in the self Grad object.
        return Grad(grads=[grad.clone() for grad in self.grads])


class Grads:
    def __init__(self, network=None, n=0, grads=None):
        # Initializes a Grads object by storing a list with n Grad objects of size of the given network, or a list of the given grads.
        if grads is not None:
            self.grads = grads
        else:
            self.grads = [Grad(network) for _ in range(n)]

    def clone(self):
        # Creates and returns a new Grads object with copies of all the Grad objects in the self Grads object.
        return Grads(grads=[grad.clone() for grad in self.grads])

    def mul(self, op):
        # Multiplies each gradient tensor element-wise by the given scalar or tensor.
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
        # Sums element-wise each gradient tensor with the given scalar, tensor or Grads object.
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
        # Computes and returns a new Grad object with the sum of gradients normalized by the number of gradients.
        grad = self.grads[0].clone()
        grad.zero()
        for g in self.grads:
            grad.add(g)
        grad.mul(1 / len(self.grads))
        return grad


def escape_float(x):
    # Replaces the decimal '.' by '\.' to avoid problems when searching for a substring with regular expressions.
    return ('%s' % x).replace('.', '\.')