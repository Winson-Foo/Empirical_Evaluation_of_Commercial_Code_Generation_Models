import torch
import numpy as np

# Create constants for device
CPU = torch.device('cpu')
CUDA = torch.device('cuda')

# Define function to select device based on gpu_id
def select_device(gpu_id):
    device = CUDA if torch.cuda.is_available() and gpu_id >= 0 else CPU
    return device

# Define function to convert input to tensor
def to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).to(device)
    return x

# Define function to create a tensor containing a sequence of integers from 0 to end-1
def range_tensor(end, device):
    return torch.arange(end).long().to(device)

# Define function to convert input to numpy array
def to_np(t):
    return t.cpu().detach().numpy()

# Define function to set random seeds for numpy and torch
def set_random_seed(seed=None):
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(int(1e6)))

# Define function to set the number of threads to 1
def set_one_thread():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

# Define function for the Huber loss function
def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

# Define function for the epsilon-greedy exploration strategy
def epsilon_greedy(epsilon, x):
    if len(x.shape) == 1:
        if np.random.rand() < epsilon:
            return np.random.randint(len(x))
        else:
            return np.argmax(x)
    elif len(x.shape) == 2:
        random_actions = np.random.randint(x.shape[1], size=x.shape[0])
        greedy_actions = np.argmax(x, axis=-1)
        dice = np.random.rand(x.shape[0])
        return np.where(dice < epsilon, random_actions, greedy_actions)

# Define function to copy the gradients of one network to another
def sync_grad(target_network, src_network):
    for target_param, src_param in zip(target_network.parameters(), src_network.parameters()):
        if src_param.grad is not None:
            target_param.grad = src_param.grad.clone()

# Define function to convert a batch of vectors into a stack of diagonal matrices
def batch_diagonal(input):
    dims = input.size()
    dims = dims + dims[-1:]
    output = torch.zeros(dims, device=input.device)
    strides = [output.stride(i) for i in range(input.dim() - 1)]
    strides.append(output.size(-1) + 1)
    output.as_strided(input.size(), strides).copy_(input)
    return output

# Define function to compute the trace of a batch of matrices
def batch_trace(input):
    i = range_tensor(input.size(-1))
    t = input[:, i, i].sum(-1).unsqueeze(-1).unsqueeze(-1)
    return t

# Define class for the Diagonal Normal distribution
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

# Define class for the Batch Categorical distribution
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

# Define class for gradients
class Grad:
    def __init__(self, model=None, grads=None):
        if grads is not None:
            self.grads = grads
        else:
            self.grads = []
            for param in model.parameters():
                self.grads.append(torch.zeros(param.data.size(), device=torch.device('cpu')))

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

    def assign(self, model):
        for grad, param in zip(self.grads, model.parameters()):
            param._grad = grad.clone()

    def zero(self):
        for grad in self.grads:
            grad.zero_()

    def clone(self):
        return Grad(grads=[grad.clone() for grad in self.grads])

# Define class for gradients of multiple models
class Grads:
    def __init__(self, model=None, n=0, grads=None):
        if grads is not None:
            self.grads = grads
        else:
            self.grads = [Grad(model) for _ in range(n)]

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
        grad.mul(1 / len(self.grads))
        return grad

# Function for escaping numbers for markdown
def escape_float(x):
    return ('%s' % x).replace('.', '\.')