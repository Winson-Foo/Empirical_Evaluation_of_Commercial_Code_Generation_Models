import torch
from nflows.transforms.conv import OneByOneConvolution
from tests.transforms.transform_test import TransformTest


class OneByOneConvolutionTestCase(TransformTest):
    def setUp(self):
        super().setUp()
        self.eps = 1e-6

    def test_forward_and_inverse_are_consistent(self):
        batch_size = 10
        num_channels, height, width = 3, 28, 28
        inputs = torch.randn(batch_size, num_channels, height, width)
        transform = OneByOneConvolution(num_channels)
        self.assert_forward_inverse_are_consistent(transform, inputs)


if __name__ == "__main__":
    unittest.main()