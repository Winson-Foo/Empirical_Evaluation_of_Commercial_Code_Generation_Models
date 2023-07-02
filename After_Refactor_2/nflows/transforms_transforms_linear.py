import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from nflows.transforms.base import Transform
from nflows.utils import torchutils
import nflows.utils.typechecks as check


class LinearCache:
    def __init__(self):
        self.weight = None
        self.inverse = None
        self.logabsdet = None

    def invalidate(self):
        self.weight = None
        self.inverse = None
        self.logabsdet = None


class Linear(Transform):
    def __init__(self, features, using_cache=False):
        if not check.is_positive_int(features):
            raise TypeError("Number of features must be a positive integer.")
        super().__init__()

        self.features = features
        self.bias = nn.Parameter(torch.zeros(features))

        self.using_cache = using_cache
        self.cache = LinearCache()

    def forward(self, inputs, context=None):
        if not self.training and self.using_cache:
            self._check_forward_cache()
            outputs = F.linear(inputs, self.cache.weight, self.bias)
            logabsdet = self.cache.logabsdet * outputs.new_ones(outputs.shape[0])
            return outputs, logabsdet
        else:
            return self.forward_no_cache(inputs)

    def _check_forward_cache(self):
        if self.cache.weight is None and self.cache.logabsdet is None:
            self.cache.weight, self.cache.logabsdet = self.weight_and_logabsdet()
        elif self.cache.weight is None:
            self.cache.weight = self.weight()
        elif self.cache.logabsdet is None:
            self.cache.logabsdet = self.logabsdet()

    def inverse(self, inputs, context=None):
        if not self.training and self.using_cache:
            self._check_inverse_cache()
            outputs = F.linear(inputs - self.bias, self.cache.inverse)
            logabsdet = (-self.cache.logabsdet) * outputs.new_ones(outputs.shape[0])
            return outputs, logabsdet
        else:
            return self.inverse_no_cache(inputs)

    def _check_inverse_cache(self):
        if self.cache.inverse is None and self.cache.logabsdet is None:
            self.cache.inverse, self.cache.logabsdet = self.weight_inverse_and_logabsdet()
        elif self.cache.inverse is None:
            self.cache.inverse = self.weight_inverse()
        elif self.cache.logabsdet is None:
            self.cache.logabsdet = self.logabsdet()

    def train(self, mode=True):
        if mode:
            self.cache.invalidate()
        return super().train(mode)

    def use_cache(self, mode=True):
        if not check.is_bool(mode):
            raise TypeError("Mode must be boolean.")
        self.using_cache = mode

    @property
    def weight(self):
        raise NotImplementedError()

    @property
    def logabsdet(self):
        raise NotImplementedError()

    def forward_no_cache(self, inputs):
        raise NotImplementedError()

    def inverse_no_cache(self, inputs):
        raise NotImplementedError()

    def weight_and_logabsdet(self):
        return self.weight(), self.logabsdet()

    def weight_inverse_and_logabsdet(self):
        return self.weight_inverse(), self.logabsdet()


class NaiveLinear(Linear):
    def __init__(self, features, orthogonal_initialization=True, using_cache=False):
        super().__init__(features, using_cache)

        if orthogonal_initialization:
            self.weight = nn.Parameter(torchutils.random_orthogonal(features))
        else:
            self.weight = nn.Parameter(torch.empty(features, features))
            stdv = 1.0 / torch.sqrt(features)
            init.uniform_(self.weight, -stdv, stdv)
  
    def forward_no_cache(self, inputs):
        batch_size = inputs.shape[0]
        outputs = F.linear(inputs, self.weight, self.bias)
        logabsdet = torchutils.logabsdet(self.weight)
        logabsdet = logabsdet * outputs.new_ones(batch_size)
        return outputs, logabsdet

    def inverse_no_cache(self, inputs):
        batch_size = inputs.shape[0]
        outputs = inputs - self.bias
        lu, lu_pivots = torch.lu(self.weight)
        outputs = torch.lu_solve(outputs.t(), lu, lu_pivots).t()
        logabsdet = -torch.sum(torch.log(torch.abs(torch.diag(lu))))
        logabsdet = logabsdet * outputs.new_ones(batch_size)
        return outputs, logabsdet

    def weight_inverse(self):
        return torch.inverse(self.weight)

    def weight_inverse_and_logabsdet(self):
        identity = torch.eye(self.features, self.features)
        lu, lu_pivots = torch.lu(self.weight)
        weight_inv = torch.lu_solve(identity, lu, lu_pivots)
        logabsdet = torch.sum(torch.log(torch.abs(torch.diag(lu))))
        return weight_inv, logabsdet