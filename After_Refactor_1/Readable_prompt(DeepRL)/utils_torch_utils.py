#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)
# Permission given to modify the code as long as you keep this
# declaration at the top
#######################################################################

import os

import numpy as np
import torch
from .config import *


def select_device(gpu_id):
    """
    Selects GPU or CPU based on availability.

    :param gpu_id: (int)
    """
    # if torch.cuda.is_available() and gpu_id >= 0:
    if gpu_id >= 0:
        Config.DEVICE = torch.device('cuda:%d' % gpu_id)
    else:
        Config.DEVICE = torch.device('cpu')


def tensor(x):
    """
    Utility function to convert numpy array to torch tensor

    :param x: array to be converted
    """
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).to(Config.DEVICE)
    return x


def range_tensor(end):
    """
    Creates a one dimensional torch tensor containing integers from 0 through (end-1)

    :param end: (int) Value representing the numbers upto which the tensor should be produced
    :return: (tensor) tensor containing integers 0 - (end - 1)
    """
    return torch.arange(end).long().to(Config.DEVICE)


def to_np(t):
    """
    Converts tensor 't' to numpy array.

    :param t: (tensor) tensor that needs to be converted to numpy array
    :return: (numpy array) tensor "t" converted to numpy array format
    """
    return t.cpu().detach().numpy()


def random_seed(seed=None):
    """
    Random seed generator that can be passed to random number generators

    :param seed: (int) Seed number
    """
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(int(1e6)))


def set_one_thread():
    """
    Function that sets the environment to use only one active thread by setting the corresponding environment variables to 1
    """
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


def huber(x, k=1.0):
    """
    Huber loss function, used for additing robustness to regression-like neural network

    :param x: (tensor) input tensor
    :param k: (float) tensor value for smoothing
    :return: smooth loss tensor
    """
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


def epsilon_greedy(epsilon, x):
    """
    Exploration strategy that used frequently with value function prediction tasks in reinforcement learning

    :param epsilon: (float) Probability that an action is picked randomly
    :param x: (numpy array) q-values for all available actions
    :return: (int) either a random action or the one with the highest q-value.
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
    Used to copy gradients calculated at target network to the original network

    :param target_network: network to which the gradients were propagated
    :param src_network: original network to which, gradients from target network are to be propagated
    """
    for param, src_param in zip(target_network.parameters(), src_network.parameters()):
        if src_param.grad is not None:
            param._grad = src_param.grad.clone()


# adapted from https://github.com/pytorch/pytorch/issues/12160
def batch_diagonal(input):
    """
    Converts stacked vector to diagonal matrices to facilitate matrix operations. (Batched Version) Batches a stack of
    vectors (batch x N) -> a stack of diagonal matrices (batch x N x N)

    :param input: (torch tensor) list of stacked vectors
    :return: (torch tensor) stack of diagonal matrices
    """
    # idea from here: https://discuss.pytorch.org/t/batch-of-diagonal-matrix/13560
    # batches a stack of vectors (batch x N) -> a stack of diagonal matrices (batch x N x N)
    # works in  2D -> 3D, should also work in higher dimensions
    # make a zero matrix, which duplicates the last dim of input
    dims = input.size()
    dims = dims + dims[-1:]
    output = torch.zeros(dims, device=input.device)
    # stride across the first dimensions, add one to get the diagonal of the last dimension
    strides = [output.stride(i) for i in range(input.dim() - 1)]  # all dimensions except the last dim
    strides.append(output.size(-1) + 1)
    # stride and copy the input to the diagonal
    output.as_strided(input.size(), strides).copy_(input)
    return output


def batch_trace(input):
    """
    Calculates trace of the matrix.

    :param input: (torch tensor) matrix
    :return: (torch tensor) traced value of the matrix
    """
    i = range_tensor(input.size(-1))
    t = input[:, i, i].sum(-1).unsqueeze(-1).unsqueeze(-1)
    return t


class DiagonalNormal:
    """
    Class that implements probability distribution over a Diagonal Gaussian
    """
    def __init__(self, mean, std):
        """
        Function to initialie DiagonalNormal instance

        :param mean: (torch tensor) tensor consisting of mean values
        :param std: (torch tensor) tensor consisting of standard deviation values
        """
        self.dist = torch.distributions.Normal(mean, std)
        self.sample = self.dist.sample

    def log_prob(self, action):
        """
        Logarithmic probability of action.

        :param action: (torch tensor) action tensor
        :return: (torch tensor) tensor containing logarithmic probabilities corresponding to parameter 'action'
        """
        return self.dist.log_prob(action).sum(-1).unsqueeze(-1)

    def entropy(self):
        """
        Function that returns the entropy of diagonally distributed Gaussian.

        :return: (torch tensor) tensor containing entropy value
        """
        return self.dist.entropy().sum(-1).unsqueeze(-1)

    def cdf(self, action):
        """
        Function that returns the cumulative distribution function of diagonally distributed Gaussian.

        :param action: (torch tensor)
        :return: (torch tensor)
        """
        return self.dist.cdf(action).prod(-1).unsqueeze(-1)


class BatchCategorical:
    """
    Class that implements probability distribution over a list of probabilities
    """
    def __init__(self, logits):
        """
        Function to initialize BatchCategorical instance .

        :param logits: (torch tensor) list of numbers
        """
        self.pre_shape = logits.size()[:-1]
        logits = logits.view(-1, logits.size(-1))
        self.dist = torch.distributions.Categorical(logits=logits)

    def log_prob(self, action):
        """
        Function that returns an array containing logarithmic probability of actions.

        :param action: (torch tensor)
        :return: (torch tensor)
        """
        log_pi = self.dist.log_prob(action.view(-1))
        log_pi = log_pi.view(action.size()[:-1] + (-1,))
        return log_pi

    def entropy(self):
        """
        Function that returns array containing entropy values of categorical distribution.

        :return: (torch tensor)
        """
        ent = self.dist.entropy()
        ent = ent.view(self.pre_shape + (-1,))
        return ent

    def sample(self, sample_shape=torch.Size([])):
        """
        Function that returns a sample from the categorical distribution.

        :param sample_shape: (torch tensor)
        :return: (torch tensor)
        """
        ret = self.dist.sample(sample_shape)
        ret = ret.view(sample_shape + self.pre_shape + (-1,))
        return ret


class Grad:
    """
    Class that encapsulates the gradient calculation
    """
    def __init__(self, network=None, grads=None):
        """
        Initializer function to instantiate Grad.

        :param network: (torch neural network)
        :param grads: (list) list of gradient values
        """
        if grads is not None:
            self.grads = grads
        else:
            self.grads = []
            for param in network.parameters():
                self.grads.append(torch.zeros(param.data.size(), device=Config.DEVICE))

    def add(self, op):
        """
        Function to add gradients.

        :param op: (one of : tensor, torch.nn.Module, Grad)
        :return: (Grad instance)
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
        Function to multiply gradients.

        :param coef: (float) factor used for the multiplication of gradients
        :return: (Grad instance)
        """
        for grad in self.grads:
            grad.mul_(coef)
        return self

    def assign(self, network):
        """
        Function to assign the gradient values to a network instance.

        :param network: (torch neural network)
        """
        for grad, param in zip(self.grads, network.parameters()):
            param._grad = grad.clone()

    def zero(self):
        """
        Function to set gradient values to 0.
        """
        for grad in self.grads:
            grad.zero_()

    def clone(self):
        """
        Function to form a clone of called Grad instance.

        :return: (Grad instance) clone of the called instance
        """
        return Grad(grads=[grad.clone() for grad in self.grads])


class Grads:
    """
    Class that encapsulates a list of Grad instances
    """
    def __init__(self, network=None, n=0, grads=None):
        """
        Initializer function for Grads.

        :param network : neural networks like MLP
        :param n: (int) no of gradients
        :param grads:  (Grads instance) list of grads
        """
        if grads is not None:
            self.grads = grads
        else:
            self.grads = [Grad(network) for _ in range(n)]

    def clone(self):
        """
        Clones Grads instance.

        :return: (Grads instance) cloned Grads instance
        """
        return Grads(grads=[grad.clone() for grad in self.grads])

    def mul(self, op):
        """
        Multiplies an instance of Grads class by another input.

        :param op: (one of : float, torch.Tensor)
        :return: (Grads instance)
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
