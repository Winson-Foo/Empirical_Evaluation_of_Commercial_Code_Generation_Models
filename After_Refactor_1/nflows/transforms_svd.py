import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from nflows.transforms.linear import Linear
from nflows.transforms.orthogonal import HouseholderSequence


class SVDLinear(Linear):
    """A linear module using the SVD decomposition for the weight matrix."""

    def __init__(
        self, features: int, num_householder: int, using_cache: bool = False, identity_init: bool = True, eps: float = 1e-3
    ):
        super().__init__(features, using_cache)

        assert num_householder % 2 == 0

        self.eps = eps
        self.orthogonal_1 = HouseholderSequence(features=features, num_transforms=num_householder)
        self.unconstrained_diagonal = nn.Parameter(torch.zeros(features))
        self.orthogonal_2 = HouseholderSequence(features=features, num_transforms=num_householder)

        self.identity_init = identity_init
        self._initialize()

    @property
    def diagonal(self) -> torch.Tensor:
        return self.eps + F.softplus(self.unconstrained_diagonal)

    @property
    def log_diagonal(self) -> torch.Tensor:
        return torch.log(self.diagonal)

    def _initialize(self) -> None:
        init.zeros_(self.bias)
        if self.identity_init:
            constant = np.log(np.exp(1 - self.eps) - 1)
            init.constant_(self.unconstrained_diagonal, constant)
        else:
            stdv = 1.0 / np.sqrt(self.features)
            init.uniform_(self.unconstrained_diagonal, -stdv, stdv)

    def forward_no_cache(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.orthogonal_2(inputs)
        outputs *= self.diagonal
        outputs, _ = self.orthogonal_1(outputs)
        outputs += self.bias

        logabsdet = self.logabsdet() * outputs.new_ones(outputs.shape[0])

        return outputs, logabsdet

    def inverse_no_cache(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs - self.bias
        outputs, _ = self.orthogonal_1.inverse(outputs)
        outputs /= self.diagonal
        outputs, _ = self.orthogonal_2.inverse(outputs)
        logabsdet = -self.logabsdet()
        logabsdet = logabsdet * outputs.new_ones(outputs.shape[0])
        return outputs, logabsdet

    def weight(self) -> torch.Tensor:
        diagonal = torch.diag(self.diagonal)
        weight, _ = self.orthogonal_2.inverse(diagonal)
        weight, _ = self.orthogonal_1(weight.t())
        return weight.t()

    def weight_inverse(self) -> torch.Tensor:
        diagonal_inv = torch.diag(torch.reciprocal(self.diagonal))
        weight_inv, _ = self.orthogonal_1(diagonal_inv)
        weight_inv, _ = self.orthogonal_2.inverse(weight_inv.t())
        return weight_inv.t()

    def logabsdet(self) -> torch.Tensor:
        return torch.sum(self.log_diagonal)