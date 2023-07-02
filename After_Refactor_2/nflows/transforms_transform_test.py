import torch
import torchtestcase

from nflows.transforms import base


class TransformTest(torchtestcase.TorchTestCase):
    """Base test for all transforms."""

    def assert_tensor_is_good(self, tensor, shape=None):
        """Assert that the tensor is valid, not containing NaN or Inf values."""
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertFalse(torch.isnan(tensor).any())
        self.assertFalse(torch.isinf(tensor).any())
        if shape is not None:
            self.assertEqual(tensor.shape, torch.Size(shape))

    def assert_forward_inverse_are_consistent(self, transform, inputs):
        """Assert that the forward and inverse transformations are consistent."""
        inverse = base.InverseTransform(transform)
        identity = base.CompositeTransform([inverse, transform])
        outputs, logabsdet = identity(inputs)

        self.assert_tensor_is_good(outputs, shape=inputs.shape)
        self.assert_tensor_is_good(logabsdet, shape=inputs.shape[:1])
        self.assertEqual(outputs, inputs)
        self.assertEqual(logabsdet, torch.zeros(inputs.shape[:1]))

    def assert_tensors_not_equal(self, first, second, msg=None):
        """Assert that two tensors are not equal."""
        if (self._eps and (first - second).abs().max().item() < self._eps) or (
            not self._eps and torch.equal(first, second)
        ):
            self._fail_with_message(msg, "The tensors are _not_ different!")