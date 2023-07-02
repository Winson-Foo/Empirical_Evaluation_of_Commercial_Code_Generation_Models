import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from nflows.transforms.linear import Linear
from nflows.transforms.orthogonal import HouseholderSequence


class SVDLinear(Linear):
    """
    A linear module using the SVD decomposition for the weight matrix.
    """

    def __init__(self, features, num_householder, using_cache=False, identity_init=True, eps=1e-3):
        super().__init__(features, using_cache)
        assert num_householder % 2 == 0

        self.eps = eps
        self.num_householder = num_householder
        self.features = features

        self.orthogonal_1 = HouseholderSequence(features=features, num_transforms=num_householder)
        self.unconstrained_diagonal = nn.Parameter(torch.zeros(features))
        self.orthogonal_2 = HouseholderSequence(features=features, num_transforms=num_householder)

        self.identity_init = identity_init
        self._initialize()

    @property
    def diagonal(self):
        """
        Returns the diagonal with a minimum value to prevent numerical instability.
        """
        return self.eps + F.softplus(self.unconstrained_diagonal)

    @property
    def log_diagonal(self):
        """
        Returns the logarithm of the diagonal entries.
        """
        return torch.log(self.diagonal)

    def _initialize(self):
        """
        Initializes the bias and diagonal matrix.
        """
        init.zeros_(self.bias)

        if self.identity_init:
            constant = np.log(np.exp(1 - self.eps) - 1)
            init.constant_(self.unconstrained_diagonal, constant)
        else:
            stdv = 1.0 / np.sqrt(self.features)
            init.uniform_(self.unconstrained_diagonal, -stdv, stdv)

    def forward_no_cache(self, inputs):
        """
        Performs forward pass without using caching.

        Cost:
            output = O(KDN)
            logabsdet = O(D)

        Args:
            inputs (torch.Tensor): The input tensor.

        Returns:
            outputs (torch.Tensor): The output tensor.
            logabsdet (torch.Tensor): The logarithm of the absolute determinant.
        """
        outputs, _ = self.orthogonal_2(inputs)
        outputs *= self.diagonal
        outputs, _ = self.orthogonal_1(outputs)
        outputs += self.bias

        logabsdet = self._logabsdet(outputs)

        return outputs, logabsdet

    def inverse_no_cache(self, inputs):
        """
        Performs inverse pass without using caching.

        Cost:
            output = O(KDN)
            logabsdet = O(D)

        Args:
            inputs (torch.Tensor): The input tensor.

        Returns:
            outputs (torch.Tensor): The output tensor.
            logabsdet (torch.Tensor): The logarithm of the absolute determinant.
        """
        outputs = inputs - self.bias
        outputs, _ = self.orthogonal_1.inverse(outputs)
        outputs /= self.diagonal
        outputs, _ = self.orthogonal_2.inverse(outputs)

        logabsdet = -self._logabsdet(outputs)

        return outputs, logabsdet

    def weight(self):
        """
        Returns the weight matrix.

        Cost:
            weight = O(KD^2)

        Returns:
            weight (torch.Tensor): The weight matrix.
        """
        diagonal = torch.diag(self.diagonal)
        weight, _ = self.orthogonal_2.inverse(diagonal)
        weight, _ = self.orthogonal_1(weight.T)

        return weight.T

    def weight_inverse(self):
        """
        Returns the inverse of the weight matrix.

        Cost:
            inverse = O(KD^2)

        Returns:
            weight_inv (torch.Tensor): The inverse of the weight matrix.
        """
        diagonal_inv = torch.diag(torch.reciprocal(self.diagonal))
        weight_inv, _ = self.orthogonal_1(diagonal_inv)
        weight_inv, _ = self.orthogonal_2.inverse(weight_inv.T)

        return weight_inv.T

    def _logabsdet(self, outputs):
        """
        Returns the logarithm of the absolute determinant.

        Cost:
            logabsdet = O(D)

        Args:
            outputs (torch.Tensor): The output tensor.

        Returns:
            logabsdet (torch.Tensor): The logarithm of the absolute determinant.
        """
        return torch.sum(self.log_diagonal)