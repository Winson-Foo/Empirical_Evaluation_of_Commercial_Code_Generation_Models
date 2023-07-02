import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from nflows.transforms.linear import Linear
from nflows.transforms.orthogonal import HouseholderSequence

class QRLinear(Linear):
    """A linear module using the QR decomposition for the weight matrix."""

    def __init__(self, features, num_householder, using_cache=False):
        super().__init__(features, using_cache)
        self.features = features
        self.num_householder = num_householder
        self._initialize_parameters()

    def _initialize_parameters(self):
        # Parameterization for R
        self.upper_indices = np.triu_indices(self.features, k=1)
        self.diag_indices = np.diag_indices(self.features)
        n_triangular_entries = ((self.features - 1) * self.features) // 2
        self.upper_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.log_upper_diag = nn.Parameter(torch.zeros(self.features))

        # Parameterization for Q
        self.orthogonal = HouseholderSequence(
            features=self.features, num_transforms=self.num_householder
        )

        self._initialize()

    def _initialize(self):
        stdv = 1.0 / np.sqrt(self.features)
        init.uniform_(self.upper_entries, -stdv, stdv)
        init.uniform_(self.log_upper_diag, -stdv, stdv)
        init.constant_(self.bias, 0.0)

    def _create_upper(self):
        upper = self.upper_entries.new_zeros(self.features, self.features)
        upper[self.upper_indices[0], self.upper_indices[1]] = self.upper_entries
        upper[self.diag_indices[0], self.diag_indices[1]] = torch.exp(
            self.log_upper_diag
        )
        return upper

    def forward_no_cache(self, inputs):
        upper = self._create_upper()

        outputs = F.linear(inputs, upper)
        outputs, _ = self.orthogonal(outputs)
        outputs += self.bias

        logabsdet = self.logabsdet() * outputs.new_ones(outputs.shape[0])

        return outputs, logabsdet

    def inverse_no_cache(self, inputs):
        upper = self._create_upper()
        outputs = inputs - self.bias
        outputs, _ = self.orthogonal.inverse(outputs)
        outputs = torch.linalg.solve_triangular(upper, outputs.t(), upper=True)
        outputs = outputs.t()
        logabsdet = -self.logabsdet()
        logabsdet = logabsdet * outputs.new_ones(outputs.shape[0])
        return outputs, logabsdet

    def weight(self):
        upper = self._create_upper()
        weight, _ = self.orthogonal(upper.t())
        return weight.t()

    def weight_inverse(self):
        upper = self._create_upper()
        identity = torch.eye(self.features, self.features)
        upper_inv = torch.linalg.solve_triangular(upper, identity, upper=True)
        weight_inv, _ = self.orthogonal(upper_inv)
        return weight_inv

    def logabsdet(self):
        return torch.sum(self.log_upper_diag)