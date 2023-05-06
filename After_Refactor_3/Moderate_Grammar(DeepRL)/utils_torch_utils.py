#######################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)
# Permission given to modify the code as long as you keep this
# declaration at the top
#######################################################

import numpy as np
import torch
import os


class Config:
    DEVICE = torch.device("cpu")
    BATCH_SIZE = 64


class DeviceSelector:
    @staticmethod
    def select(gpu_id):
        if gpu_id >= 0:
            Config.DEVICE = torch.device(f"cuda:{gpu_id}")
        else:
            Config.DEVICE = torch.device("cpu")


class Tensor:
    @staticmethod
    def from_numpy_or_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        x = np.asarray(x, dtype=np.float32)
        x = torch.from_numpy(x).to(Config.DEVICE)
        return x

    @staticmethod
    def range(end):
        return torch.arange(end).long().to(Config.DEVICE)

    @staticmethod
    def to_numpy(tensor):
        return tensor.cpu().detach().numpy()


class RandomSeed:
    @staticmethod
    def set(seed=None):
        np.random.seed(seed)
        torch.manual_seed(np.random.randint(int(1e6)))


class ThreadPool:
    @staticmethod
    def set_threads():
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)


class HuberLoss:
    @staticmethod
    def compute(x, k=1.0):
        return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


class EpsilonGreedy:
    @staticmethod
    def select_action(epsilon, x):
        if len(x.shape) == 1:
            return np.random.randint(len(x)) if np.random.rand() < epsilon else np.argmax(x)
        elif len(x.shape) == 2:
            random_actions = np.random.randint(x.shape[1], size=x.shape[0])
            greedy_actions = np.argmax(x, axis=-1)
            dice = np.random.rand(x.shape[0])
            return np.where(dice < epsilon, random_actions, greedy_actions)


class ModelSync:
    @staticmethod
    def copy_grads(src_model, dest_model):
        for param, src_param in zip(dest_model.parameters(), src_model.parameters()):
            if src_param.grad is not None:
                param.grad = src_param.grad.clone()


class BatchDiagonal:
    @staticmethod
    def create(input):
        # batches a stack of vectors (batch x N) -> a stack of diagonal matrices (batch x N x N)
        # works in  2D -> 3D, should also work in higher dimensions
        dims = input.size()
        dims = dims + dims[-1:]
        output = torch.zeros(dims, device=input.device)
        strides = [output.stride(i) for i in range(input.dim() - 1)]
        strides.append(output.size(-1) + 1)
        output.as_strided(input.size(), strides).copy_(input)
        return output


class BatchTrace:
    @staticmethod
    def compute(input):
        i = Tensor.range(input.size(-1))
        t = input[:, i, i].sum(-1).unsqueeze(-1).unsqueeze(-1)
        return t


class DiagonalNormal:
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
    def __init__(self, network=None, grads=None):
        if grads is not None:
            self.grads = grads
        else:
            self.grads = []
            for param in network.parameters():
                self.grads.append(torch.zeros(param.data.size(), device=Config.DEVICE))

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
            grad.zero_()

    def zero(self):
        for grad in self.grads:
            grad.zero_()

    def clone(self):
        return Grad(grads=[grad.clone() for grad in self.grads])


class Grads:
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
                grad.add_(op_grad)
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
        grad.mul(1 / len(self.grads))
        return grad


class FloatEscaper:
    @staticmethod
    def escape(x):
        return str(x).replace(".", "\.")


# Usage example
if __name__ == "__main__":
    RandomSeed.set(0)
    Tensor.to_numpy(torch.tensor([1.0, 2.0, 3.0], device="cpu"))  # [1.0, 2.0, 3.0]