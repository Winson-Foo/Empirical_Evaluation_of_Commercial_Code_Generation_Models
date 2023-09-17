import unittest
import torch

from nflows.transforms import nonlinearities as nl
from nflows.transforms import standard
from nflows.transforms.base import InputOutsideDomain
from tests.transforms.transform_test import TransformTest


# Nonlinearities Tests
class NonlinearitiesTest(TransformTest):
    """Tests for the nonlinearities transforms."""

    def test_forward(self):
        """Test the forward method of the nonlinearities transforms."""
        batch_size = 10
        shape = [5, 10, 15]
        inputs = torch.rand(batch_size, *shape)
        transforms = [
            nl.Exp(),
            nl.Tanh(),
            nl.LogTanh(),
            nl.LeakyReLU(),
            nl.Sigmoid(),
            nl.Logit(),
            nl.CompositeCDFTransform(nl.Sigmoid(), standard.IdentityTransform()),
        ]
        for transform in transforms:
            with self.subTest(transform=transform):
                outputs, logabsdet = transform(inputs)
                self.assert_tensor_is_good(outputs, [batch_size] + shape)
                self.assert_tensor_is_good(logabsdet, [batch_size])

    def test_inverse(self):
        """Test the inverse method of the nonlinearities transforms."""
        batch_size = 10
        shape = [5, 10, 15]
        inputs = torch.rand(batch_size, *shape)
        transforms = [
            nl.Exp(),
            nl.Tanh(),
            nl.LogTanh(),
            nl.LeakyReLU(),
            nl.Sigmoid(),
            nl.Logit(),
            nl.CompositeCDFTransform(nl.Sigmoid(), standard.IdentityTransform()),
        ]
        for transform in transforms:
            with self.subTest(transform=transform):
                outputs, logabsdet = transform.inverse(inputs)
                self.assert_tensor_is_good(outputs, [batch_size] + shape)
                self.assert_tensor_is_good(logabsdet, [batch_size])

    def test_forward_inverse_are_consistent(self):
        """Test that the forward and inverse methods are consistent for the nonlinearities transforms."""
        batch_size = 10
        shape = [5, 10, 15]
        inputs = torch.rand(batch_size, *shape)
        transforms = [
            nl.Exp(),
            nl.Tanh(),
            nl.LogTanh(),
            nl.LeakyReLU(),
            nl.Sigmoid(),
            nl.Logit(),
            nl.CompositeCDFTransform(nl.Sigmoid(), standard.IdentityTransform()),
        ]
        self.eps = 1e-3
        for transform in transforms:
            with self.subTest(transform=transform):
                self.assert_forward_inverse_are_consistent(transform, inputs)


# Invertible Nonlinearities Tests
class ExpTest(TransformTest):
    """Tests for the Exp transform."""

    def test_raises_domain_exception(self):
        """Test that Exp.transform.inverse raises InputOutsideDomain exception."""
        shape = [2, 3, 4]
        transform = nl.Exp()
        for value in [-1.0, 0.0]:
            with self.assertRaises(InputOutsideDomain):
                inputs = torch.full(shape, value)
                transform.inverse(inputs)


class TanhTest(TransformTest):
    """Tests for the Tanh transform."""

    def test_raises_domain_exception(self):
        """Test that Tanh.transform.inverse raises InputOutsideDomain exception."""
        shape = [2, 3, 4]
        transform = nl.Tanh()
        for value in [-2.0, -1.0, 1.0, 2.0]:
            with self.assertRaises(InputOutsideDomain):
                inputs = torch.full(shape, value)
                transform.inverse(inputs)


# Piecewise CDF Tests
class TestPiecewiseCDF(TransformTest):
    """Tests for the PiecewiseCDF transforms."""

    def setUp(self):
        """Set up the test case."""
        self.shape = [2, 3, 4]
        self.batch_size = 10
        self.transforms = [
            nl.PiecewiseLinearCDF(self.shape),
            nl.PiecewiseQuadraticCDF(self.shape),
            nl.PiecewiseCubicCDF(self.shape),
            nl.PiecewiseRationalQuadraticCDF(self.shape),
        ]

    def test_raises_domain_exception(self):
        """Test that the transforms.forward method raises InputOutsideDomain exception."""
        for transform in self.transforms:
            with self.subTest(transform=transform):
                for value in [-1.0, -0.1, 1.1, 2.0]:
                    with self.assertRaises(InputOutsideDomain):
                        inputs = torch.full([self.batch_size] + self.shape, value)
                        transform.forward(inputs)

    def test_zeros_to_zeros(self):
        """Test that the transforms.forward method maps zeros to zeros."""
        for transform in self.transforms:
            with self.subTest(transform=transform):
                inputs = torch.zeros(self.batch_size, *self.shape)
                outputs, _ = transform(inputs)
                self.eps = 1e-5
                self.assertEqual(outputs, inputs)

    def test_ones_to_ones(self):
        """Test that the transforms.forward method maps ones to ones."""
        for transform in self.transforms:
            with self.subTest(transform=transform):
                inputs = torch.ones(self.batch_size, *self.shape)
                outputs, _ = transform(inputs)
                self.eps = 1e-5
                self.assertEqual(outputs, inputs)

    def test_forward_inverse_are_consistent(self):
        """Test that the transforms.forward and transforms.inverse methods are consistent."""
        for transform in self.transforms:
            with self.subTest(transform=transform):
                inputs = torch.rand(self.batch_size, *self.shape)
                self.eps = 1e-3
                self.assert_forward_inverse_are_consistent(transform, inputs)


class TestUnconstrainedPiecewiseCDF(TransformTest):
    """Tests for the unconstrained PiecewiseCDF transforms."""

    def test_forward_inverse_are_consistent(self):
        """Test that the transforms.forward and transforms.inverse methods are consistent."""
        shape = [2, 3, 4]
        batch_size = 10
        transforms = [
            nl.PiecewiseLinearCDF(shape, tails="linear"),
            nl.PiecewiseQuadraticCDF(shape, tails="linear"),
            nl.PiecewiseCubicCDF(shape, tails="linear"),
            nl.PiecewiseRationalQuadraticCDF(shape, tails="linear"),
        ]

        for transform in transforms:
            with self.subTest(transform=transform):
                inputs = 3 * torch.randn(batch_size, *shape)
                self.eps = 1e-3
                self.assert_forward_inverse_are_consistent(transform, inputs)


class LogitTest(TransformTest):
    """Tests for the Logit transform."""

    def test_forward_zero_and_one(self):
        """Test that the forward method maps zeros and ones properly."""
        batch_size = 10
        shape = [5, 10, 15]
        inputs = torch.cat(
            [torch.zeros(batch_size // 2, *shape), torch.ones(batch_size // 2, *shape)]
        )

        transform = nl.Logit()
        outputs, logabsdet = transform(inputs)

        self.assert_tensor_is_good(outputs)
        self.assert_tensor_is_good(logabsdet)


if __name__ == "__main__":
    unittest.main()