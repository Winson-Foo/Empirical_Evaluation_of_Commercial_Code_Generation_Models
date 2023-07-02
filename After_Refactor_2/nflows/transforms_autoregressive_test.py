import unittest
import torch
from parameterized import parameterized
from nflows.transforms import autoregressive
from tests.transforms.transform_test import TransformTest


class MaskedAffineAutoregressiveTransformTest(TransformTest):
    @parameterized.expand([
        (False, False),
        (False, True),
        (True, False)
    ])
    def test_forward(self, use_residual_blocks, random_mask):
        batch_size = 10
        num_features = 20
        inputs = torch.randn(batch_size, num_features)
        transform = self._initialize_transform(num_features, use_residual_blocks, random_mask)
        outputs, logabsdet = transform(inputs)
        self.assert_tensor_is_good(outputs, [batch_size, num_features])
        self.assert_tensor_is_good(logabsdet, [batch_size])

    @parameterized.expand([
        (False, False),
        (False, True),
        (True, False)
    ])
    def test_inverse(self, use_residual_blocks, random_mask):
        batch_size = 10
        num_features = 20
        inputs = torch.randn(batch_size, num_features)
        transform = self._initialize_transform(num_features, use_residual_blocks, random_mask)
        outputs, logabsdet = transform.inverse(inputs)
        self.assert_tensor_is_good(outputs, [batch_size, num_features])
        self.assert_tensor_is_good(logabsdet, [batch_size])

    @parameterized.expand([
        (False, False),
        (False, True),
        (True, False)
    ])
    def test_forward_inverse_are_consistent(self, use_residual_blocks, random_mask):
        batch_size = 10
        num_features = 20
        inputs = torch.randn(batch_size, num_features)
        self.eps = 1e-6
        transform = self._initialize_transform(num_features, use_residual_blocks, random_mask)
        self.assert_forward_inverse_are_consistent(transform, inputs)

    def _initialize_transform(self, num_features, use_residual_blocks, random_mask):
        return autoregressive.MaskedAffineAutoregressiveTransform(
            features=num_features,
            hidden_features=30,
            num_blocks=5,
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask
        )


class MaskedPiecewiseLinearAutoregressiveTranformTest(TransformTest):
    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        num_features = 20
        inputs = torch.rand(batch_size, num_features)
        self.eps = 1e-3
        transform = autoregressive.MaskedPiecewiseLinearAutoregressiveTransform(
            num_bins=10,
            features=num_features,
            hidden_features=30,
            num_blocks=5,
            use_residual_blocks=True
        )
        self.assert_forward_inverse_are_consistent(transform, inputs)


class MaskedPiecewiseQuadraticAutoregressiveTranformTest(TransformTest):
    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        num_features = 20
        inputs = torch.rand(batch_size, num_features)
        self.eps = 1e-4
        transform = autoregressive.MaskedPiecewiseQuadraticAutoregressiveTransform(
            num_bins=10,
            features=num_features,
            hidden_features=30,
            num_blocks=5,
            use_residual_blocks=True
        )
        self.assert_forward_inverse_are_consistent(transform, inputs)


class MaskedUMNNAutoregressiveTranformTest(TransformTest):
    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        num_features = 20
        inputs = torch.rand(batch_size, num_features)
        self.eps = 1e-4
        transform = autoregressive.MaskedUMNNAutoregressiveTransform(
            cond_size=10,
            features=num_features,
            hidden_features=30,
            num_blocks=5,
            use_residual_blocks=True
        )
        self.assert_forward_inverse_are_consistent(transform, inputs)


class MaskedPiecewiseCubicAutoregressiveTranformTest(TransformTest):
    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        num_features = 20
        inputs = torch.rand(batch_size, num_features)
        self.eps = 1e-3
        transform = autoregressive.MaskedPiecewiseCubicAutoregressiveTransform(
            num_bins=10,
            features=num_features,
            hidden_features=30,
            num_blocks=5,
            use_residual_blocks=True
        )
        self.assert_forward_inverse_are_consistent(transform, inputs)


if __name__ == "__main__":
    unittest.main()