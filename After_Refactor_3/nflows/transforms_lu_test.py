import unittest
import torch
from nflows.transforms import lu
from tests.transforms.transform_test import TransformTest

class LULinearTest(TransformTest):
    def setUp(self):
        self.features = 3
        self.transform = lu.LULinear(features=self.features)
        self.weight, self.weight_inverse = self.transform._create_lower_upper()
        self.logabsdet = torch.log(torch.abs(torch.det(self.weight)))
        self.eps = 1e-5

    def test_forward_no_cache(self):
        batch_size = 10
        input_tensor = torch.randn(batch_size, self.features)
        output_tensor, logabsdet = self.transform.forward_no_cache(input_tensor)

        expected_output_tensor = input_tensor @ self.weight.t() + self.transform.bias
        expected_logabsdet = self.logabsdet.item() * torch.ones(batch_size)

        self.assert_tensor_is_good(output_tensor, [batch_size, self.features])
        self.assert_tensor_is_good(logabsdet, [batch_size])
        self.assertEqual(output_tensor, expected_output_tensor)
        self.assertEqual(logabsdet, expected_logabsdet)

    def test_inverse_no_cache(self):
        batch_size = 10
        input_tensor = torch.randn(batch_size, self.features)
        output_tensor, logabsdet = self.transform.inverse_no_cache(input_tensor)

        expected_output_tensor = (input_tensor - self.transform.bias) @ self.weight_inverse.t()
        expected_logabsdet = -self.logabsdet.item() * torch.ones(batch_size)

        self.assert_tensor_is_good(output_tensor, [batch_size, self.features])
        self.assert_tensor_is_good(logabsdet, [batch_size])
        self.assertEqual(output_tensor, expected_output_tensor)
        self.assertEqual(logabsdet, expected_logabsdet)

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
        self.assert_tensor_is_good(logabsdet, [1])
        self.assertEqual(logabsdet, self.logabsdet)

    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        input_tensor = torch.randn(batch_size, self.features)
        self.assert_forward_inverse_are_consistent(self.transform, input_tensor)

if __name__ == "__main__":
    unittest.main()