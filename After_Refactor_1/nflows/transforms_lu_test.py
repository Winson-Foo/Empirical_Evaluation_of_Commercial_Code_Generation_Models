import unittest
import torch
from nflows.transforms.lu import LULinear
from nflows.utils import torchutils
from tests.transforms.transform_test import TransformTest


class LULinearTransformTest(TransformTest):
    def setUp(self):
        self.features = 3
        self.transform = LULinear(features=self.features)
        self.weight, self.weight_inverse, self.logabsdet = self._calculate_parameters()

    def _calculate_parameters(self):
        lower, upper = self.transform._create_lower_upper()
        weight = lower @ upper
        weight_inverse = torch.inverse(weight)
        logabsdet = torchutils.logabsdet(weight)
        return weight, weight_inverse, logabsdet

    def _assert_output(self, output, output_ref, logabsdet, logabsdet_ref):
        self.assert_tensor_is_good(output, [batch_size, self.features])
        self.assert_tensor_is_good(logabsdet, [batch_size])
        self.assertEqual(output, output_ref)
        self.assertEqual(logabsdet, logabsdet_ref)

    def test_forward_no_cache(self):
        batch_size = 10
        inputs = torch.randn(batch_size, self.features)
        output_ref = inputs @ self.weight.t() + self.transform.bias
        logabsdet_ref = torch.full([batch_size], self.logabsdet.item())

        output, logabsdet = self.transform.forward_no_cache(inputs)
        self._assert_output(output, output_ref, logabsdet, logabsdet_ref)

    def test_inverse_no_cache(self):
        batch_size = 10
        inputs = torch.randn(batch_size, self.features)
        output_ref = (inputs - self.transform.bias) @ self.weight_inverse.t()
        logabsdet_ref = torch.full([batch_size], -self.logabsdet.item())

        output, logabsdet = self.transform.inverse_no_cache(inputs)
        self._assert_output(output, output_ref, logabsdet, logabsdet_ref)

    def test_weight(self):
        weight = self.transform.weight()
        self.assert_tensor_is_good(weight, [self.features, self.features])
        self.assertEqual(weight, self.weight)

    def test_weight_inverse(self):
        weight_inverse = self.transform.weight_inverse()
        self.assert_tensor_is_good(weight_inverse, [self.features, self.features])
        self.assertEqual(weight_inverse, self.weight_inverse)

    def test_logabsdet(self):
        logabsdet = self.transform.logabsdet()
        self.assert_tensor_is_good(logabsdet, [])
        self.assertEqual(logabsdet, self.logabsdet)

    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        inputs = torch.randn(batch_size, self.features)
        self.assert_forward_inverse_are_consistent(self.transform, inputs)


if __name__ == "__main__":
    unittest.main()