import torch
import torchtestcase
from nflows.transforms import base

class TransformTest(torchtestcase.TorchTestCase):
    """Base test for all transforms."""

    def check_tensor(self, tensor: torch.Tensor, shape: torch.Size = None):
        assert isinstance(tensor, torch.Tensor)
        assert not torch.isnan(tensor).any()
        assert not torch.isinf(tensor).any()
        if shape:
            assert tensor.shape == shape

    def check_forward_inverse_consistency(self, transform: base.Transform, inputs: torch.Tensor):
        inverse = base.InverseTransform(transform)
        identity = base.CompositeTransform([inverse, transform])
        outputs, logabsdet = identity(inputs)

        self.check_tensor(outputs, shape=inputs.shape)
        self.check_tensor(logabsdet, shape=inputs.shape[:1])
        assert torch.equal(outputs, inputs)
        assert torch.equal(logabsdet, torch.zeros(inputs.shape[:1]))

    def check_inequality(self, first: torch.Tensor, second: torch.Tensor, msg: str = None):
        if (
            (self._eps and (first - second).abs().max().item() < self._eps)
            or (not self._eps and torch.equal(first, second))
        ):
            self._fail_with_message(msg, "The tensors are _not_ different!")