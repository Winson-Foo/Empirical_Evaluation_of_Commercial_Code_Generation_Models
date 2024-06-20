import unittest
import torch
from nflows.transforms import autoregressive
from tests.transforms.transform_test import TransformTest

class AutoregressiveTransformTestCase(TransformTest):
    def setUp(self):
        self.batch_size = 10
        self.features = 20
        self.inputs = torch.randn(self.batch_size, self.features)
        self.eps = None

    def _setup_transform(self, transform):
        transform = autoregressive.MaskedAffineAutoregressiveTransform(
            features=self.features,
            hidden_features=30,
            num_blocks=5,
            use_residual_blocks=transform["use_residual_blocks"],
            random_mask=transform["random_mask"],
        )
        return transform

    def test_forward(self):
        configurations = [
            {"use_residual_blocks": False, "random_mask": False},
            {"use_residual_blocks": False, "random_mask": True},
            {"use_residual_blocks": True, "random_mask": False},
        ]

        for config in configurations:
            with self.subTest(config=config):
                transform = self._setup_transform(config)
                outputs, logabsdet = transform(self.inputs)
                self.assert_tensor_is_good(outputs, [self.batch_size, self.features])
                self.assert_tensor_is_good(logabsdet, [self.batch_size])

    def test_inverse(self):
        configurations = [
            {"use_residual_blocks": False, "random_mask": False},
            {"use_residual_blocks": False, "random_mask": True},
            {"use_residual_blocks": True, "random_mask": False},
        ]

        for config in configurations:
            with self.subTest(config=config):
                transform = self._setup_transform(config)
                outputs, logabsdet = transform.inverse(self.inputs)
                self.assert_tensor_is_good(outputs, [self.batch_size, self.features])
                self.assert_tensor_is_good(logabsdet, [self.batch_size])

    def test_forward_inverse_are_consistent(self):
        configurations = [
            {"use_residual_blocks": False, "random_mask": False},
            {"use_residual_blocks": False, "random_mask": True},
            {"use_residual_blocks": True, "random_mask": False},
        ]
        
        self.eps = 1e-6

        for config in configurations:
            with self.subTest(config=config):
                transform = self._setup_transform(config)
                self.assert_forward_inverse_are_consistent(transform, self.inputs)


class MaskedPiecewiseLinearAutoregressiveTranformTest(TransformTest):
    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        features = 20
        inputs = torch.rand(batch_size, features)
        self.eps = 1e-3

        transform = autoregressive.MaskedPiecewiseLinearAutoregressiveTransform(
            num_bins=10,
            features=features,
            hidden_features=30,
            num_blocks=5,
            use_residual_blocks=True,
        )

        self.assert_forward_inverse_are_consistent(transform, inputs)


class MaskedPiecewiseQuadraticAutoregressiveTranformTest(TransformTest):
    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        features = 20
        inputs = torch.rand(batch_size, features)
        self.eps = 1e-4

        transform = autoregressive.MaskedPiecewiseQuadraticAutoregressiveTransform(
            num_bins=10,
            features=features,
            hidden_features=30,
            num_blocks=5,
            use_residual_blocks=True,
        )

        self.assert_forward_inverse_are_consistent(transform, inputs)


class MaskedUMNNAutoregressiveTranformTest(TransformTest):
    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        features = 20
        inputs = torch.rand(batch_size, features)
        self.eps = 1e-4

        transform = autoregressive.MaskedUMNNAutoregressiveTransform(
            cond_size=10,
            features=features,
            hidden_features=30,
            num_blocks=5,
            use_residual_blocks=True,
        )

        self.assert_forward_inverse_are_consistent(transform, inputs)


class MaskedPiecewiseCubicAutoregressiveTranformTest(TransformTest):
    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        features = 20
        inputs = torch.rand(batch_size, features)
        self.eps = 1e-3

        transform = autoregressive.MaskedPiecewiseCubicAutoregressiveTransform(
            num_bins=10,
            features=features,
            hidden_features=30,
            num_blocks=5,
            use_residual_blocks=True,
        )

        self.assert_forward_inverse_are_consistent(transform, inputs)


if __name__ == "__main__":
    unittest.main()