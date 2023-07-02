import numpy as np
import torch
from torch import nn

from nflows.distributions.base import Distribution
from nflows.utils import torchutils


class StandardNormal(Distribution):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, shape: tuple):
        super().__init__()
        self._shape = torch.Size(shape)
        self.register_buffer("_log_z",
                             torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi),
                                          dtype=torch.float64),
                             persistent=False)

    def _log_prob(self, inputs: torch.Tensor, context: torch.Tensor or None) -> torch.Tensor:
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                f"Expected input of shape {self._shape}, got {inputs.shape[1:]}"
            )
        neg_energy = -0.5 * torchutils.sum_except_batch(inputs ** 2, num_batch_dims=1)
        return neg_energy - self._log_z

    def _sample(self, num_samples: int, context: torch.Tensor or None) -> torch.Tensor:
        if context is None:
            return torch.randn(num_samples, *self._shape, device=self._log_z.device)
        else:
            context_size = context.shape[0]
            samples = torch.randn(context_size * num_samples, *self._shape, device=context.device)
            return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context: torch.Tensor or None) -> torch.Tensor:
        if context is None:
            return self._log_z.new_zeros(self._shape)
        else:
            return torch.zeros(context.shape[0], *self._shape)


class ConditionalDiagonalNormal(Distribution):
    """A diagonal multivariate Normal whose parameters are functions of a context."""

    def __init__(self, shape: tuple, context_encoder: callable = None):
        super().__init__()
        self._shape = torch.Size(shape)
        if context_encoder is None:
            self._context_encoder = lambda x: x
        else:
            self._context_encoder = context_encoder
        self.register_buffer("_log_z",
                             torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi),
                                          dtype=torch.float64),
                             persistent=False)

    def _compute_params(self, context: torch.Tensor) -> tuple:
        if context is None:
            raise ValueError("Context can't be None.")

        params = self._context_encoder(context)
        if params.shape[-1] % 2 != 0:
            raise RuntimeError(
                "The context encoder must return a tensor whose last dimension is even."
            )
        if params.shape[0] != context.shape[0]:
            raise RuntimeError(
                "The batch dimension of the parameters is inconsistent with the input."
            )

        split = params.shape[-1] // 2
        means = params[..., :split].reshape(params.shape[0], *self._shape)
        log_stds = params[..., split:].reshape(params.shape[0], *self._shape)
        return means, log_stds

    def _log_prob(self, inputs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                f"Expected input of shape {self._shape}, got {inputs.shape[1:]}"
            )

        means, log_stds = self._compute_params(context)
        norm_inputs = (inputs - means) * torch.exp(-log_stds)
        log_prob = -0.5 * torchutils.sum_except_batch(norm_inputs ** 2, num_batch_dims=1)
        log_prob -= torchutils.sum_except_batch(log_stds, num_batch_dims=1)
        log_prob -= self._log_z
        return log_prob

    def _sample(self, num_samples: int, context: torch.Tensor) -> torch.Tensor:
        means, log_stds = self._compute_params(context)
        stds = torch.exp(log_stds)
        means = torchutils.repeat_rows(means, num_samples)
        stds = torchutils.repeat_rows(stds, num_samples)

        context_size = context.shape[0]
        noise = torch.randn(context_size * num_samples, *self._shape, device=means.device)
        samples = means + stds * noise
        return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context: torch.Tensor) -> torch.Tensor:
        means, _ = self._compute_params(context)
        return means


class DiagonalNormal(Distribution):
    """A diagonal multivariate Normal with trainable parameters."""

    def __init__(self, shape: tuple):
        super().__init__()
        self._shape = torch.Size(shape)
        self.mean_ = nn.Parameter(torch.zeros(shape).reshape(1, -1))
        self.log_std_ = nn.Parameter(torch.zeros(shape).reshape(1, -1))
        self.register_buffer("_log_z",
                             torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi),
                                          dtype=torch.float64),
                             persistent=False)

    def _log_prob(self, inputs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                f"Expected input of shape {self._shape}, got {inputs.shape[1:]}"
            )

        means = self.mean_
        log_stds = self.log_std_

        norm_inputs = (inputs - means) * torch.exp(-log_stds)
        log_prob = -0.5 * torchutils.sum_except_batch(norm_inputs ** 2, num_batch_dims=1)
        log_prob -= torchutils.sum_except_batch(log_stds, num_batch_dims=1)
        log_prob -= self._log_z
        return log_prob

    def _sample(self, num_samples: int, context: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def _mean(self, context: torch.Tensor) -> torch.Tensor:
        return self.mean