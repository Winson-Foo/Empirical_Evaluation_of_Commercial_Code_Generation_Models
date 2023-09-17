import unittest
import torch
from transform import SqueezeTransform

class SqueezeTransformTest(unittest.TestCase):
    def setUp(self):
        self.transform = SqueezeTransform()

    def test_forward(self):
        batch_size = 10
        for shape in [[32, 4, 4], [16, 8, 8]]:
            with self.subTest(shape=shape):
                channels, height, width = shape
                inputs = torch.randn(batch_size, channels, height, width)
                outputs, logabsdet = self.transform.forward(inputs)
                self.assert_tensor_is_good(outputs, [batch_size, channels * 4, height // 2, width // 2])
                self.assert_tensor_is_good(logabsdet, [batch_size])
                self.assertEqual(logabsdet, torch.zeros(batch_size))

    def test_forward_values(self):
        inputs = torch.arange(1, 17, 1).long().view(1, 1, 4, 4)
        outputs, _ = self.transform.forward(inputs)
        # rest of the test method

    def test_forward_wrong_shape(self):
        batch_size = 10
        for shape in [[32, 3, 3], [32, 5, 5], [32, 4]]:
            with self.subTest(shape=shape):
                inputs = torch.randn(batch_size, *shape)
                with self.assertRaises(ValueError):
                    self.transform.forward(inputs)

    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        for shape in [[32, 4, 4], [16, 8, 8]]:
            with self.subTest(shape=shape):
                channels, height, width = shape
                inputs = torch.randn(batch_size, channels, height, width)
                self.assert_forward_inverse_are_consistent(self.transform, inputs)

    def test_inverse_wrong_shape(self):
        batch_size = 10
        for shape in [[3, 4, 4], [33, 4, 4], [32, 4]]:
            with self.subTest(shape=shape):
                inputs = torch.randn(batch_size, *shape)
                with self.assertRaises(ValueError):
                    self.transform.inverse(inputs)

if __name__ == "__main__":
    unittest.main()