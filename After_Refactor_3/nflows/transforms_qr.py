import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from nflows.transforms.linear import Linear
from nflows.transforms.orthogonal import HouseholderSequence

class QRLinear(Linear):
    def __init__(self, features, num_householder, using_cache=False):
        super().__init__(features, using_cache)
        self._initialize(features, num_householder)

    def _initialize(self, features, num_householder):
        self._initialize_parameters(features)
        self._initialize_orthogonal(features, num_householder)

    def _initialize_parameters(self, features):
        self._initialize_upper_entries(features)
        self._initialize_log_upper_diag(features)
        self._initialize_bias()

    def _initialize_upper_entries(self, features):
        upper_indices = np.triu_indices(features, k=1)
        n_triangular_entries = ((features - 1) * features) // 2
        self.upper_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        stdv = 1.0 / np.sqrt(features)
        init.uniform_(self.upper_entries, -stdv, stdv)

    def _initialize_log_upper_diag(self, features):
        self.log_upper_diag = nn.Parameter(torch.zeros(features))
        stdv = 1.0 / np.sqrt(features)
        init.uniform_(self.log_upper_diag, -stdv, stdv)

    def _initialize_bias(self):
        init.constant_(self.bias, 0.0)

    def _initialize_orthogonal(self, features, num_householder):
        self.orthogonal = HouseholderSequence(features=features, num_transforms=num_householder)

    def forward_no_cache(self, inputs):
        upper = self._create_upper()

        outputs = F.linear(inputs, upper)
        outputs, _ = self.orthogonal(outputs)
        outputs += self.bias

        logabsdet = self.logabsdet() * outputs.new_ones(outputs.shape[0])

        return outputs, logabsdet

    # Rest of the code...