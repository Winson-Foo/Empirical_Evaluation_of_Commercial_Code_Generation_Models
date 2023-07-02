import unittest
import torch

from nflows.transforms.conv import OneByOneConvolution
from tests.transforms.transform_test import TransformTest


class OneByOneConvolutionTest(TransformTest):
    def test_forward_and_inverse_are_consistent(self):
        batch_size = 10
        channels, height, width = 3, 28, 28
        inputs = torch.randn(batch_size, channels, height, width)
        transform = OneByOneConvolution(channels)
        self.eps = 1e-6
        self.assert_forward_inverse_are_consistent(transform, inputs)


def run_unit_tests():
    unittest.main()


if __name__ == "__main__":
    run_unit_tests()