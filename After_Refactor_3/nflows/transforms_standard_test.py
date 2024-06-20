import unittest
import numpy as np
import torch
from nflows.transforms.standard import IdentityTransform, AffineScalarTransform
from tests.transforms.transform_test import TransformTest

class IdentityTransformTest(TransformTest):
    def test_forward(self):
        batch_size = 10
        shape = [2, 3, 4]
        inputs = torch.randn(batch_size, *shape)
        transform = IdentityTransform()
        outputs, logabsdet = transform(inputs)
        self.assertEqual(outputs, inputs)
        self.assertEqual(logabsdet, torch.zeros(batch_size))

    def test_inverse(self):
        batch_size = 10
        shape = [2, 3, 4]
        inputs = torch.randn(batch_size, *shape)
        transform = IdentityTransform()
        outputs, logabsdet = transform.inverse(inputs)
        self.assertEqual(outputs, inputs)
        self.assertEqual(logabsdet, torch.zeros(batch_size))

    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        shape = [2, 3, 4]
        inputs = torch.randn(batch_size, *shape)
        transform = IdentityTransform()
        self.assert_forward_inverse_are_consistent(transform, inputs)

class AffineScalarTransformTest(TransformTest):
    def test_forward(self):
        batch_size = 10
        shape = [2, 3, 4]
        inputs = torch.randn(batch_size, *shape)

        def test_case(scale, shift, true_outputs, true_logabsdet):
            with self.subTest(scale=scale, shift=shift):
                transform = AffineScalarTransform(scale=scale, shift=shift)
                outputs, logabsdet = transform(inputs)
                self.assertEqual(outputs, true_outputs)
                self.assertEqual(
                    logabsdet, torch.full([batch_size], true_logabsdet * np.prod(shape))
                )

        self.eps = 1e-6
        test_case(None, 2.0, inputs + 2.0, 0.)
        test_case(2.0, None, inputs * 2.0, np.log(2.0))
        test_case(2.0, 2.0, inputs * 2.0 + 2.0, np.log(2.0))
        test_case(-1.0, None, -inputs, 0.0)
        test_case(-2.0, 2.0, inputs * -2.0 + 2.0, np.log(2.0))

    def test_inverse(self):
        batch_size = 10
        shape = [2, 3, 4]
        inputs = torch.randn(batch_size, *shape)

        def test_case(scale, shift, true_outputs, true_logabsdet):
            with self.subTest(scale=scale, shift=shift):
                transform = AffineScalarTransform(scale=scale, shift=shift)
                outputs, logabsdet = transform.inverse(inputs)
                self.assertEqual(outputs, true_outputs)
                self.assertEqual(
                    logabsdet, torch.full([batch_size], true_logabsdet * np.prod(shape))
                )

        self.eps = 1e-6
        test_case(None, 2.0, inputs - 2.0, 0.)
        test_case(2.0, None, inputs / 2.0, -np.log(2.0))
        test_case(2.0, 2.0, (inputs - 2.0) / 2.0, -np.log(2.0))
        test_case(-1.0, None, -inputs, 0.0)
        test_case(-2.0, 2.0, (inputs - 2.0) / -2.0, -np.log(2.0))

    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        shape = [2, 3, 4]
        inputs = torch.randn(batch_size, *shape)

        def test_case(scale, shift):
            transform = AffineScalarTransform(scale=scale, shift=shift)
            self.assert_forward_inverse_are_consistent(transform, inputs)

        self.eps = 1e-6
        test_case(None, 2.0)
        test_case(2.0, None)
        test_case(2.0, 2.0)
        test_case(-1.0, None)
        test_case(-2.0, 2.0)
        
    def test_raises_value_error(self):
        def test_case(shift):
            with self.assertRaises(ValueError):
                transform = AffineScalarTransform(scale=0.0, shift=shift)

        test_case(None)

if __name__ == "__main__":
    unittest.main()