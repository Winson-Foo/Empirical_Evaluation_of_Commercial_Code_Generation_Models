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

        # Parameterization for R
        self.upper_indices = np.triu_indices(features, k=1)
        self.diag_indices = np.diag_indices(features)
        n_triangular_entries = ((features - 1) * features) // 2
        self.upper_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.log_upper_diag = nn.Parameter(torch.zeros(features))

        # Parameterization for Q
        self.orthogonal = HouseholderSequence(
            features=features, num_transforms=num_householder
        )

        self._initialize()

    def _initialize(self):
        std_deviation = 1.0 / np.sqrt(self.features)
        init.uniform_(self.upper_entries, -std_deviation, std_deviation)
        init.uniform_(self.log_upper_diag, -std_deviation, std_deviation)
        init.constant_(self.bias, 0.0)

    def _create_upper(self):
        upper_matrix = self.upper_entries.new_zeros(self.features, self.features)
        upper_matrix[self.upper_indices[0], self.upper_indices[1]] = self.upper_entries
        upper_matrix[self.diag_indices[0], self.diag_indices[1]] = torch.exp(
            self.log_upper_diag
        )
        return upper_matrix

    def forward_no_cache(self, inputs):
        upper_matrix = self._create_upper()

        outputs = F.linear(inputs, upper_matrix)
        outputs, _ = self.orthogonal(outputs)
        outputs += self.bias

        logabsdet = self.compute_logabsdet(outputs)

        return outputs, logabsdet

    def inverse_no_cache(self, inputs):
        upper_matrix = self._create_upper()
        outputs = inputs - self.bias
        outputs, _ = self.orthogonal.inverse(outputs)
        outputs = torch.linalg.solve_triangular(upper_matrix, outputs.t(), upper=True)
        outputs = outputs.t()
        logabsdet = -self.compute_logabsdet(outputs)
        return outputs, logabsdet

    def weight(self):
        upper_matrix = self._create_upper()
        weight, _ = self.orthogonal(upper_matrix.t())
        return weight.t()

    def weight_inverse(self):
        upper_matrix = self._create_upper()
        identity_matrix = torch.eye(self.features, self.features)
        upper_inverse_matrix = torch.linalg.solve_triangular(upper_matrix, identity_matrix, upper=True)
        weight_inverse, _ = self.orthogonal(upper_inverse_matrix)
        return weight_inverse

    def compute_logabsdet(self, outputs):
        return torch.sum(self.log_upper_diag)