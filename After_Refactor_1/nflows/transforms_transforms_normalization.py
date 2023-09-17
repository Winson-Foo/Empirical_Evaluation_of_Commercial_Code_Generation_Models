import numpy as np
import torch
from torch import nn

from nflows.transforms.base import InverseNotAvailable, Transform
import nflows.utils.typechecks as check


class BatchNorm(Transform):
    """Transform that performs batch normalization.

    Limitations:
        * It works only for 1-dim inputs.
        * Inverse is not available in training mode, only in eval mode.
    """

    def __init__(self, features, eps=1e-5, momentum=0.1, affine=True):
        if not check.is_positive_int(features):
            raise TypeError('Number of features must be a positive integer.')
        super().__init__()

        self.momentum = momentum
        self.eps = eps
        constant = np.log(np.exp(1 - eps) - 1)
        self.unconstrained_weight = nn.Parameter(constant * torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

        self.register_buffer("running_mean", torch.zeros(features))
        self.register_buffer("running_var", torch.zeros(features))

    @property
    def weight(self):
        return torch.nn.functional.softplus(self.unconstrained_weight) + self.eps

    def forward(self, inputs, context=None):
        if self.training:
            mean, var = inputs.mean(0), inputs.var(0)
            self.running_mean.mul_(1 - self.momentum).add_(mean.detach() * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(var.detach() * self.momentum)
        else:
            mean, var = self.running_mean, self.running_var

        outputs = self.weight * ((inputs - mean) / torch.sqrt((var + self.eps))) + self.bias

        logabsdet_ = torch.log(self.weight) - 0.5 * torch.log(var + self.eps)
        logabsdet = torch.sum(logabsdet_) * inputs.new_ones(inputs.shape[0])

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if self.training:
            raise InverseNotAvailable(
                'Batch norm inverse is only available in eval mode, not in training mode.')
        outputs = (
            torch.sqrt(self.running_var + self.eps)
            * ((inputs - self.bias) / self.weight)
            + self.running_mean
        )
        logabsdet_ = -torch.log(self.weight) + 0.5 * torch.log(self.running_var + self.eps)
        logabsdet = torch.sum(logabsdet_) * inputs.new_ones(inputs.shape[0])

        return outputs, logabsdet


class ActNorm(Transform):
    def __init__(self, features):
        """
        Transform that performs activation normalization. Works for 2D and 4D inputs. For 4D
        inputs (images) normalization is performed per-channel, assuming BxCxHxW input shape.

        Reference:
        > D. Kingma et. al., Glow: Generative flow with invertible 1x1 convolutions, NeurIPS 2018.
        """
        if not check.is_positive_int(features):
            raise TypeError("Number of features must be a positive integer.")
        super().__init__()

        self.log_scale = nn.Parameter(torch.zeros(features))
        self.shift = nn.Parameter(torch.zeros(features))

    @property
    def scale(self):
        return torch.exp(self.log_scale)

    def forward(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError("Expecting inputs to be a 2D or a 4D tensor.")

        scale, shift = self.scale, self.shift
        outputs = scale * inputs + shift

        if inputs.dim() == 4:
            logabsdet = inputs.size(2) * inputs.size(3) * torch.sum(self.log_scale) * torch.ones(inputs.size(0))
        else:
            logabsdet = torch.sum(self.log_scale) * torch.ones(inputs.size(0))

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError("Expecting inputs to be a 2D or a 4D tensor.")

        scale, shift = self.scale, self.shift
        outputs = (inputs - shift) / scale

        if inputs.dim() == 4:
            logabsdet = -inputs.size(2) * inputs.size(3) * torch.sum(self.log_scale) * torch.ones(inputs.size(0))
        else:
            logabsdet = -torch.sum(self.log_scale) * torch.ones(inputs.size(0))

        return outputs, logabsdet


class InverseNotAvailable(Exception):
    pass