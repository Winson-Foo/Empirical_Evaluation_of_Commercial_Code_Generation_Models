import unittest

import numpy as np
import torch

from nflows.transforms import standard
from tests.transforms.transform_test import TransformTest

class IdentityTransformTest(TransformTest):
    def test_forward(self):
        batch_size = 10
        input_shape = [2, 3, 4]
        inputs = torch.randn(batch_size, *input_shape)
        transform = standard.IdentityTransform()
        outputs, logabsdet = transform(inputs)

        self.assert_tensor_is_good(outputs, [batch_size] + input_shape)
        self.assert_tensor_is_good(logabsdet, [batch_size])
        self.assertEqual(outputs, inputs)
        self.assertEqual(logabsdet, torch.zeros(batch_size))

    def test_inverse(self):
        batch_size = 10
        input_shape = [2, 3, 4]
        inputs = torch.randn(batch_size, *input_shape)
        transform = standard.IdentityTransform()
        outputs, logabsdet = transform.inverse(inputs)

        self.assert_tensor_is_good(outputs, [batch_size] + input_shape)
        self.assert_tensor_is_good(logabsdet, [batch_size])
        self.assertEqual(outputs, inputs)
        self.assertEqual(logabsdet, torch.zeros(batch_size))

    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        input_shape = [2, 3, 4]
        inputs = torch.randn(batch_size, *input_shape)
        transform = standard.IdentityTransform()
        
        self.assert_forward_inverse_are_consistent(transform, inputs)


class AffineScalarTransformTest(TransformTest):
    def test_forward(self):
        batch_size = 10
        input_shape = [2, 3, 4]
        inputs = torch.randn(batch_size, *input_shape)
        self.eps = 1e-6

        def test_case(scale, shift, expected_outputs, expected_logabsdet):
            with self.subTest(scale=scale, shift=shift):
                transform = standard.AffineScalarTransform(scale=scale, shift=shift)
                outputs, logabsdet = transform(inputs)
                self.assert_tensor_is_good(outputs, [batch_size] + input_shape)
                self.assert_tensor_is_good(logabsdet, [batch_size])
                self.assertEqual(outputs, expected_outputs)
                self.assertEqual(logabsdet, torch.full([batch_size], expected_logabsdet * np.prod(input_shape)))

        test_case(None, 2.0, inputs + 2.0, 0.)
        test_case(2.0, None, inputs * 2.0, np.log(2.0))
        test_case(2.0, 2.0, inputs * 2.0 + 2.0, np.log(2.0))
        test_case(-1.0, None, -inputs, 0.0)
        test_case(-2.0, 2.0, inputs * -2.0 + 2.0, np.log(2.0))

    def test_inverse(self):
        batch_size = 10
        input_shape = [2, 3, 4]
        inputs = torch.randn(batch_size, *input_shape)
        self.eps = 1e-6

        def test_case(scale, shift, expected_outputs, expected_logabsdet):
            with self.subTest(scale=scale, shift=shift):
                transform = standard.AffineScalarTransform(scale=scale, shift=shift)
                outputs, logabsdet = transform.inverse(inputs)
                self.assert_tensor_is_good(outputs, [batch_size] + input_shape)
                self.assert_tensor_is_good(logabsdet, [batch_size])
                self.assertEqual(outputs, expected_outputs)
                self.assertEqual(logabsdet, torch.full([batch_size], expected_logabsdet * np.prod(input_shape)))

        test_case(None, 2.0, inputs - 2.0, 0.)
        test_case(2.0, None, inputs / 2.0, -np.log(2.0))
        test_case(2.0, 2.0, (inputs - 2.0) / 2.0, -np.log(2.0))
        test_case(-1.0, None, -inputs, 0.0)
        test_case(-2.0, 2.0, (inputs - 2.0) / -2.0, -np.log(2.0))

    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        input_shape = [2, 3, 4]
        inputs = torch.randn(batch_size, *input_shape)
        self.eps = 1e-6

        def test_case(scale, shift):
            transform = standard.AffineScalarTransform(scale=scale, shift=shift)
            self.assert_forward_inverse_are_consistent(transform, inputs)

        test_case(None, 2.0)
        test_case(2.0, None)
        test_case(2.0, 2.0)
        test_case(-1.0, None)
        test_case(-2.0, 2.0)
        
    def test_raises_value_error(self):
        def test_case(shift):
            with self.assertRaises(ValueError):
                transform = standard.AffineTransform(scale=0.0, shift=shift)
            
        test_case(None)


if __name__ == "__main__":
    unittest.main()