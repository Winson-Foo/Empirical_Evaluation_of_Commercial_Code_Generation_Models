import torch
import numpy as np

from nflows.utils import typechecks as check


def tile(x: torch.Tensor, n: int) -> torch.Tensor:
    check.is_positive_int(n, "Argument 'n' must be a positive integer.")
    x_ = x.reshape(-1)
    x_ = x_.repeat(n)
    x_ = x_.reshape(n, -1)
    x_ = x_.transpose(1, 0)
    x_ = x_.reshape(-1)
    return x_


def sum_except_batch(x: torch.Tensor, num_batch_dims: int = 1) -> torch.Tensor:
    check.is_nonnegative_int(num_batch_dims, "Number of batch dimensions must be a non-negative integer.")
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


def split_leading_dim(x: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    new_shape = shape + x.shape[1:]
    return torch.reshape(x, new_shape)


def merge_leading_dims(x: torch.Tensor, num_dims: int) -> torch.Tensor:
    check.is_positive_int(num_dims, "Number of leading dims must be a positive integer.")
    if num_dims > x.dim():
        raise ValueError("Number of leading dims can't be greater than total number of dims.")
    new_shape = torch.Size([-1]) + x.shape[num_dims:]
    return torch.reshape(x, new_shape)


def repeat_rows(x: torch.Tensor, num_reps: int) -> torch.Tensor:
    check.is_positive_int(num_reps, "Number of repetitions must be a positive integer.")
    shape = x.shape
    x = x.unsqueeze(1)
    x = x.expand(shape[0], num_reps, *shape[1:])
    return merge_leading_dims(x, num_dims=2)


def tensor2numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def logabsdet(x: torch.Tensor) -> torch.Tensor:
    _, res = torch.slogdet(x)
    return res


def random_orthogonal(size: int) -> torch.Tensor:
    x = torch.randn(size, size)
    q, _ = torch.qr(x)
    return q


def get_num_parameters(model: torch.nn.Module) -> int:
    num_parameters = 0
    for parameter in model.parameters():
        num_parameters += torch.numel(parameter)
    return num_parameters


def create_alternating_binary_mask(features: int, even: bool = True) -> torch.Tensor:
    mask = torch.zeros(features, dtype=torch.uint8)
    start = 0 if even else 1
    mask[start::2] += 1
    return mask


def create_mid_split_binary_mask(features: int) -> torch.Tensor:
    mask = torch.zeros(features, dtype=torch.uint8)
    midpoint = features // 2 if features % 2 == 0 else features // 2 + 1
    mask[:midpoint] += 1
    return mask


def create_random_binary_mask(features: int) -> torch.Tensor:
    mask = torch.zeros(features, dtype=torch.uint8)
    weights = torch.ones(features, dtype=torch.float32)
    num_samples = features // 2 if features % 2 == 0 else features // 2 + 1
    indices = torch.multinomial(input=weights, num_samples=num_samples, replacement=False)
    mask[indices] += 1
    return mask


def searchsorted(bin_locations: torch.Tensor, inputs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def cbrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.exp(torch.log(torch.abs(x)) / 3.0)


def get_temperature(max_value: int, bound: float = 1 - 1e-3) -> float:
    max_value = torch.Tensor([max_value])
    bound = torch.Tensor([bound])
    temperature = min(-(1 / max_value) * (torch.log1p(-bound) - torch.log(bound)), 1)
    return temperature.item()


def gaussian_kde_log_eval(samples: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    N, D = samples.shape[0], samples.shape[-1]
    std = N ** (-1 / (D + 4))
    precision = (1 / (std ** 2)) * torch.eye(D)
    a = query - samples
    b = a @ precision
    c = -0.5 * torch.sum(a * b, dim=-1)
    d = -np.log(N) - (D / 2) * np.log(2 * np.pi) - D * np.log(std)
    c += d
    return torch.logsumexp(c, dim=-1)