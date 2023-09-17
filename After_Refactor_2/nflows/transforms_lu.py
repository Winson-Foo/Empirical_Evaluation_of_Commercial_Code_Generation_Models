# linear.py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


class Linear(nn.Module):
    """A base class for linear transforms."""

    def __init__(self, features, using_cache=False):
        super().__init__()

        self.features = features
        self.using_cache = using_cache

        self.weight = nn.Parameter(torch.Tensor(features, features))
        self.bias = nn.Parameter(torch.Tensor(features))

        self.reset_parameters()

        if self.using_cache:
            self.register_buffer('cached_weight', None)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        """Apply the linear transform."""
        outputs = F.linear(inputs, self.weight, self.bias)
        logabsdet = torch.slogdet(self.weight)[1] * torch.ones(inputs.shape[0])
        return outputs, logabsdet

    def inverse(self, inputs):
        """Apply the inverse linear transform."""
        outputs = inputs - self.bias
        outputs = torch.solve(outputs.t(), self.weight).solution
        outputs = outputs.t()

        logabsdet = -torch.slogdet(self.weight)[1] * torch.ones(inputs.shape[0])

        return outputs, logabsdet


class LULinear(Linear):
    """A linear transform where we parameterize the LU decomposition of the weights."""

    def __init__(self, features, using_cache=False, identity_init=True, eps=1e-3):
        super().__init__(features, using_cache)

        self.eps = eps

        self.lower_indices = np.tril_indices(features, k=-1)
        self.upper_indices = np.triu_indices(features, k=1)
        self.diag_indices = np.diag_indices(features)

        n_triangular_entries = ((features - 1) * features) // 2

        self.lower_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.upper_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.unconstrained_upper_diag = nn.Parameter(torch.zeros(features))

        self._initialize(identity_init)

    def _initialize(self, identity_init):
        init.zeros_(self.bias)

        if identity_init:
            init.zeros_(self.lower_entries)
            init.zeros_(self.upper_entries)
            constant = np.log(np.exp(1 - self.eps) - 1)
            init.constant_(self.unconstrained_upper_diag, constant)
        else:
            stdv = 1.0 / np.sqrt(self.features)
            init.uniform_(self.lower_entries, -stdv, stdv)
            init.uniform_(self.upper_entries, -stdv, stdv)
            init.uniform_(self.unconstrained_upper_diag, -stdv, stdv)

    def _create_lower_upper(self):
        lower = self.lower_entries.new_zeros(self.features, self.features)
        lower[self.lower_indices[0], self.lower_indices[1]] = self.lower_entries
        # The diagonal of L is taken to be all-ones without loss of generality.
        lower[self.diag_indices[0], self.diag_indices[1]] = 1.0

        upper = self.upper_entries.new_zeros(self.features, self.features)
        upper[self.upper_indices[0], self.upper_indices[1]] = self.upper_entries
        upper[self.diag_indices[0], self.diag_indices[1]] = self.upper_diag

        return lower, upper

    def forward(self, inputs):
        """Apply the LU linear transform."""
        lower, upper = self._create_lower_upper()
        outputs = F.linear(inputs, upper)
        outputs = F.linear(outputs, lower, self.bias)
        logabsdet = self.logabsdet() * inputs.new_ones(outputs.shape[0])
        return outputs, logabsdet

    def inverse(self, inputs):
        """Apply the inverse LU linear transform."""
        lower, upper = self._create_lower_upper()
        outputs = inputs - self.bias
        outputs = torch.linalg.solve_triangular(
            lower, outputs.t(), upper=False, unitriangular=True
        )
        outputs = torch.linalg.solve_triangular(
            upper, outputs, upper=True, unitriangular=False
        )
        outputs = outputs.t()

        logabsdet = -self.logabsdet()
        logabsdet = logabsdet * inputs.new_ones(outputs.shape[0])

        return outputs, logabsdet

    def weight(self):
        """Get the weight matrix."""
        lower, upper = self._create_lower_upper()
        return lower @ upper

    def weight_inverse(self):
        """Get the inverse of the weight matrix."""
        lower, upper = self._create_lower_upper()
        identity = torch.eye(
            self.features, self.features, device=self.lower_entries.device)
        lower_inverse = torch.linalg.solve_triangular(
            lower, identity, upper=False, unitriangular=True
        )
        weight_inverse = torch.linalg.solve_triangular(
            upper, lower_inverse, upper=True, unitriangular=False
        )
        return weight_inverse

    @property
    def upper_diag(self):
        """Get the upper diagonal of the weight matrix."""
        return F.softplus(self.unconstrained_upper_diag) + self.eps

    def logabsdet(self):
        """Compute the log absolute determinant of the weight matrix."""
        return torch.sum(torch.log(self.upper_diag))