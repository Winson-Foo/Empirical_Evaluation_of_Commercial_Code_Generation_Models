import unittest
import torch
from nflows.transforms.reshape import SqueezeTransform
from tests.transforms.transform_test import TransformTest


class SqueezeTransformTest(TransformTest):
    """
    Unit tests for the SqueezeTransform class.
    """

    def setUp(self):
        self.squeeze_transform = SqueezeTransform()

    def test_forward(self):
        """
        Test the forward method of SqueezeTransform.
        """
        batch_size = 10
        test_shapes = [[32, 4, 4], [16, 8, 8]]

        for shape in test_shapes:
            with self.subTest(shape=shape):
                c, h, w = shape
                inputs = torch.randn(batch_size, c, h, w)
                outputs, logabsdet = self.squeeze_transform(inputs)

                expected_output_shape = [batch_size, c * 4, h // 2, w // 2]
                self.assert_tensor_is_good(outputs, expected_output_shape)
                self.assert_tensor_is_good(logabsdet, [batch_size])
                self.assertEqual(logabsdet, torch.zeros(batch_size))

    def test_forward_values(self):
        """
        Test the forward method of SqueezeTransform with specific input values.
        """
        inputs = torch.arange(1, 17, 1).long().view(1, 1, 4, 4)
        outputs, _ = self.squeeze_transform(inputs)

        def assert_channel_equal(channel, expected_values):
            self.assertEqual(outputs[0, channel, ...], torch.LongTensor(expected_values))

        assert_channel_equal(0, [[1, 3], [9, 11]])
        assert_channel_equal(1, [[2, 4], [10, 12]])
        assert_channel_equal(2, [[5, 7], [13, 15]])
        assert_channel_equal(3, [[6, 8], [14, 16]])

    def test_forward_wrong_shape(self):
        """
        Test the forward method of SqueezeTransform with wrong input shapes.
        """
        batch_size = 10
        test_shapes = [[32, 3, 3], [32, 5, 5], [32, 4]]

        for shape in test_shapes:
            with self.subTest(shape=shape):
                inputs = torch.randn(batch_size, *shape)
                with self.assertRaises(ValueError):
                    self.squeeze_transform(inputs)

    def test_inverse_wrong_shape(self):
        """
        Test the inverse method of SqueezeTransform with wrong input shapes.
        """
        batch_size = 10
        test_shapes = [[3, 4, 4], [33, 4, 4], [32, 4]]

        for shape in test_shapes:
            with self.subTest(shape=shape):
                inputs = torch.randn(batch_size, *shape)
                with self.assertRaises(ValueError):
                    self.squeeze_transform.inverse(inputs)

    def test_forward_inverse_are_consistent(self):
        """
        Test the forward and inverse methods of SqueezeTransform for consistency.
        """
        batch_size = 10
        test_shapes = [[32, 4, 4], [16, 8, 8]]

        for shape in test_shapes:
            with self.subTest(shape=shape):
                c, h, w = shape
                inputs = torch.randn(batch_size, c, h, w)
                self.assert_forward_inverse_are_consistent(self.squeeze_transform, inputs)


if __name__ == "__main__":
    unittest.main()