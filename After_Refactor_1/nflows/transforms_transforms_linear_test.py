import unittest
import torch

from nflows.transforms.linear import Linear, NaiveLinear


class LinearTest(unittest.TestCase):
    def setUp(self):
        features = 5
        batch_size = 10

        weight = torch.randn(features, features)
        inverse = torch.randn(features, features)
        logabsdet = torch.randn(1)
        self.transform = Linear(features)
        self.transform.bias.data = torch.randn(features)

        self.inputs = torch.randn(batch_size, features)
        self.outputs_fwd = self.inputs @ weight.t() + self.transform.bias
        self.outputs_inv = (self.inputs - self.transform.bias) @ inverse.t()
        self.logabsdet_fwd = logabsdet * torch.ones(batch_size)
        self.logabsdet_inv = (-logabsdet) * torch.ones(batch_size)

    def test_forward_default(self):
        outputs, logabsdet = self.transform.forward(self.inputs)

        self.assertEqual(outputs, self.outputs_fwd)
        self.assertEqual(logabsdet, self.logabsdet_fwd)

    def test_inverse_default(self):
        outputs, logabsdet = self.transform.inverse(self.inputs)

        self.assertEqual(outputs, self.outputs_inv)
        self.assertEqual(logabsdet, self.logabsdet_inv)


class NaiveLinearTest(unittest.TestCase):
    def setUp(self):
        self.features = 3
        self.transform = NaiveLinear(features=self.features)

        self.weight = self.transform.weight()
        self.weight_inverse = self.transform.weight_inverse()
        self.logabsdet = self.transform.logabsdet()

    def test_forward_no_cache(self):
        batch_size = 10
        inputs = torch.randn(batch_size, self.features)
        outputs, logabsdet = self.transform.forward_no_cache(inputs)

        outputs_ref = inputs @ self.weight.t() + self.transform.bias
        logabsdet_ref = torch.full([batch_size], self.logabsdet.item())

        self.assertEqual(outputs, outputs_ref)
        self.assertEqual(logabsdet, logabsdet_ref)

    def test_inverse_no_cache(self):
        batch_size = 10
        inputs = torch.randn(batch_size, self.features)
        outputs, logabsdet = self.transform.inverse_no_cache(inputs)

        outputs_ref = (inputs - self.transform.bias) @ self.weight_inverse.t()
        logabsdet_ref = torch.full([batch_size], -self.logabsdet.item())

        self.assertEqual(outputs, outputs_ref)
        self.assertEqual(logabsdet, logabsdet_ref)


if __name__ == "__main__":
    unittest.main()