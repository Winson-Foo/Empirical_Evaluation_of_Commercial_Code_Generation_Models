from .transforms import Linear
from nflows.utils import torchutils


class NaiveLinear(Linear):
    def __init__(self, features, bias=None):
        super().__init__(features, bias)
        self._weight = torch.randn(features, features)

    def forward_no_cache(self, inputs):
        outputs = inputs @ self.weight().t() + self.bias
        logabsdet = torch.full([inputs.shape[0]], self.logabsdet().item())
        return outputs, logabsdet

    def inverse_no_cache(self, inputs):
        outputs = (inputs - self.bias) @ self.weight_inverse().t()
        logabsdet = torch.full([inputs.shape[0]], -self.logabsdet().item())
        return outputs, logabsdet

    def weight(self):
        return self._weight

    def weight_inverse(self):
        return torch.inverse(self._weight)

    def logabsdet(self):
        return torchutils.logabsdet(self._weight)