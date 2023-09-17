import unittest
import torch
from nflows.transforms.conv import OneByOneConvolution
from tests.transforms.transform_test import TransformTest


class OneByOneConvolutionTest(TransformTest):
    
    def test_forward_and_inverse(self):
        batch_size = 10
        input_channels, height, width = 3, 28, 28
        inputs = torch.randn(batch_size, input_channels, height, width)
        transform = OneByOneConvolution(input_channels)
        self.eps = 1e-6
        self.assert_forward_inverse_consistency(transform, inputs)

    
def main():
    unittest.main()


if __name__ == "__main__":
    main()