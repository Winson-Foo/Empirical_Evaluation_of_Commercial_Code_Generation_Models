import torch
from nflows.transforms import base


class TransformTester:
    """Base test class for all transforms."""

    @staticmethod
    def assert_tensor_valid(tensor: torch.Tensor, shape: torch.Size = None) -> None:
        assert isinstance(tensor, torch.Tensor)
        assert not torch.isnan(tensor).any()
        assert not torch.isinf(tensor).any()
        if shape is not None:
            assert tensor.shape == torch.Size(shape)

    @staticmethod
    def assert_forward_inverse_consistent(transform, inputs):
        inverse_transform = base.InverseTransform(transform)
        identity_transform = base.CompositeTransform([inverse_transform, transform])
        outputs, logabsdet = identity_transform(inputs)

        assert_tensor_valid(outputs, shape=inputs.shape)
        assert_tensor_valid(logabsdet, shape=inputs.shape[:1])
        assert outputs == inputs
        assert logabsdet == torch.zeros(inputs.shape[:1])


class TransformTest(torchtestcase.TorchTestCase, TransformTester):
    """Test class for all transforms."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eps = 1e-6

    def assertNotEqual(self, first, second, msg=None):
        if (self._eps and (first - second).abs().max().item() < self._eps) or (
            not self._eps and torch.equal(first, second)
        ):
            self._fail_with_message(msg, "The tensors are _not_ different!")