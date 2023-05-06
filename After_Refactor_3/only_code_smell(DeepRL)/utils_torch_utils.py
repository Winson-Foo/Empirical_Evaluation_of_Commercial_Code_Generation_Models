from .config import *
import torch
import os

def select_device(gpu_id):
	"""Set the device to run on using the given GPU ID, or fallback to CPU if negative."""
    Config.DEVICE = torch.device('cuda:%d' % (gpu_id)) if gpu_id >= 0 else torch.device('cpu')

def tensor(x):
	"""Convert the given tensor-like object to a Tensor and move to the current device."""
    return torch.as_tensor(x, dtype=torch.float32, device=Config.DEVICE)

def range_tensor(size):
	"""Create a Tensor representing a range from 0 to size-1."""
    return torch.arange(size, dtype=torch.long, device=Config.DEVICE)

def to_np(t):
	"""Move the given Tensor to the CPU and convert to a numpy array."""
    return t.detach().cpu().numpy()

def random_seed(seed=None):
	"""Seed the random number generators for NumPy and PyTorch."""
	np.random.seed(seed)
	torch.manual_seed(np.random.randint(int(1e6)))

def set_one_thread():
	"""Set the environment variables required to limit MKL/OpenMP to a single thread."""
	os.environ['OMP_NUM_THREADS'] = '1'
	os.environ['MKL_NUM_THREADS'] = '1'
	torch.set_num_threads(1)

def huber(x, k=1.0):
	"""Compute the Huber loss with the given threshold `k`."""
	return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

def epsilon_greedy(epsilon, x):
	"""Select a random action with probability `epsilon`, or the greedy action otherwise."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if np.random.rand() < epsilon:
        return np.random.randint(len(x))
    else:
        return np.argmax(x)

def sync_grad(target_network, src_network):
	"""Copy the gradients from the given source network to the target network."""
    for target_param, src_param in zip(target_network.parameters(), src_network.parameters()):
        if src_param.grad is not None:
            target_param.grad = src_param.grad.clone()

def batch_diagonal(input):
	"""Convert a batch of vectors to a batch of diagonal matrices."""
    batch_size, num_elements = input.shape[:-1]
    output = torch.zeros(batch_size, num_elements, num_elements, device=input.device)
    output.as_strided((batch_size, num_elements), (num_elements * (output.stride(0),) + output.stride(-1),)).copy_(input)
	return output

def batch_trace(input):
	"""Compute the batch trace of the given Tensor."""
    batch_size, num_rows, num_cols = input.shape
    index = torch.arange(num_rows, device=input.device)
    return input[:, index, index].sum(dim=-1, keepdim=True)

class DiagonalNormal:
	"""Utility class for a diagonal multivariate normal distribution."""
    def __init__(self, mean, std):
        self.dist = torch.distributions.Normal(mean, std)

    def log_prob(self, action):
        return self.dist.log_prob(action).sum(dim=-1, keepdim=True)

    def entropy(self):
        return self.dist.entropy().sum(dim=-1, keepdim=True)

    def cdf(self, action):
        return self.dist.cdf(action).prod(dim=-1, keepdim=True)

class BatchCategorical:
	"""Utility class for a batch of categorical distributions."""
	def __init__(self, logits):
		batch_size, num_actions = logits.shape
		self.dist = torch.distributions.Categorical(logits=logits.view(batch_size, -1))

	def log_prob(self, action):
		return self.dist.log_prob(action.view(-1)).view(action.shape[:-1] + (1,))

	def entropy(self):
		return self.dist.entropy().view(self.pre_shape + (-1,))

	def sample(self, sample_shape=torch.Size([])):
		batch_size = self.pre_shape[0]
		return self.dist.sample(sample_shape).view(sample_shape + (batch_size,) + self.pre_shape[1:] + (-1,))

class Grad:
	"""Wrapper for a collection of gradients corresponding to a network's parameters."""
    def __init__(self, network):
        self.grads = [torch.zeros_like(param.data, device=Config.DEVICE) for param in network.parameters()]

    def add(self, other):
        if isinstance(other, torch.Tensor):
            for grad in self.grads:
                grad.add_(other.reshape(grad.shape))
        elif isinstance(other, Grad):
            for grad, other_grad in zip(self.grads, other.grads):
                grad.add_(other_grad)
        elif isinstance(other, torch.nn.Module):
            for grad, param in zip(self.grads, other.parameters()):
                if param.grad is not None:
                    grad.add_(param.grad)
        return self

    def mul(self, factor):
        for grad in self.grads:
            grad.mul_(factor)
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
	"""Wrapper for a collection of gradients corresponding to multiple networks' parameters."""
	def __init__(self, network, n):
		self.grads = [Grad(network) for _ in range(n)]

	def clone(self):
    	return Grads(grads=[grad.clone() for grad in self.grads])

	def mul(self, factor):
		if isinstance(factor, int) or isinstance(factor, float):
			for grad in self.grads:
				grad.mul(factor)
		elif isinstance(factor, torch.Tensor):
			assert len(factor.shape) == 1, "Factor tensor must be 1-D."
			for i, grad in enumerate(self.grads):
				grad.mul(factor[i])
		else:
			raise ValueError(f"Unsupported type for factor: {type(factor)}")

	def add(self, other):
		if isinstance(other, Grads):
			assert len(other.grads) == len(self.grads), "Incompatible length for 'other'."
			for grad, other_grad in zip(self.grads, other.grads):
				grad.add(other_grad)
		else:
			raise ValueError(f"Unsupported type for 'other': {type(other)}")

	def mean(self):
		output_grads = [torch.stack([grads.grads[j] for grads in self.grads]).mean(dim=0).unsqueeze(0) for j in range(len(self.grads[0].grads))]
		return Grad(network=None, grads=output_grads)
