import unittest
import torch
from nflows.transforms.qr import QRLinear
from nflows.utils.torchutils import logabsdet
from tests.transforms.transform_test import TransformTest

class QRLinearTest(TransformTest):
    def setUp(self):
        self.num_features = 3
        self.num_householders = 4
        self.transform = QRLinear(features=self.num_features, num_householder=self.num_householders)

        upper = self.transform.create_upper()
        orthogonal = self.transform.orthogonal_matrix()
        self.weight = orthogonal @ upper
        self.weight_inverse = torch.inverse(self.weight)
        self.logabsdet = logabsdet(self.weight)

        self.eps = 1e-5

    def test_forward_no_cache(self):
        batch_size = 10
        inputs = torch.randn(batch_size, self.num_features)
        outputs, logabsdet = self.transform.forward_no_cache(inputs)

        outputs_ref = torch.matmul(inputs, self.weight.t()) + self.transform.bias
        logabsdet_ref = torch.full([batch_size], self.logabsdet.item())

        self.assert_tensor_is_good(outputs, [batch_size, self.num_features])
        self.assert_tensor_is_good(logabsdet, [batch_size])

        self.assertEqual(outputs, outputs_ref)
        self.assertEqual(logabsdet, logabsdet_ref)

    def test_inverse_no_cache(self):
        batch_size = 10
        inputs = torch.randn(batch_size, self.num_features)
        outputs, logabsdet = self.transform.inverse_no_cache(inputs)

        outputs_ref = (inputs - self.transform.bias) @ self.weight_inverse.t()
        logabsdet_ref = torch.full([batch_size], -self.logabsdet.item())

        self.assert_tensor_is_good(outputs, [batch_size, self.num_features])
        self.assert_tensor_is_good(logabsdet, [batch_size])

        self.assertEqual(outputs, outputs_ref)
        self.assertEqual(logabsdet, logabsdet_ref)

    def test_weight(self):
        weight = self.transform.weight()
        self.assert_tensor_is_good(weight, [self.num_features, self.num_features])
        self.assertEqual(weight, self.weight)

    def test_weight_inverse(self):
        weight_inverse = self.transform.weight_inverse()
        self.assert_tensor_is_good(weight_inverse, [self.num_features, self.num_features])
        self.assertEqual(weight_inverse, self.weight_inverse)

    def test_logabsdet(self):
        logabsdet = self.transform.logabsdet()
        self.assert_tensor_is_good(logabsdet, [])
        self.assertEqual(logabsdet, self.logabsdet)

    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        inputs = torch.randn(batch_size, self.num_features)
        self.assert_forward_inverse_are_consistent(self.transform, inputs)


if __name__ == "__main__":
    unittest.main()