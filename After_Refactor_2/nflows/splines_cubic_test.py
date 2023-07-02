import torch
import torchtestcase
from nflows.transforms import splines

class CubicSplineTest(torchtestcase.TorchTestCase):
    def setUp(self):
        self.num_bins = 10
        self.shape = [2, 3, 4]
        self.unnormalized_widths = torch.randn(*self.shape, self.num_bins)
        self.unnormalized_heights = torch.randn(*self.shape, self.num_bins)
        self.unnorm_derivatives_left = torch.randn(*self.shape, 1)
        self.unnorm_derivatives_right = torch.randn(*self.shape, 1)
        self.tail_bound = 1.0

    def test_forward_inverse_are_consistent(self):
        def call_spline_fn(inputs, inverse=False):
            return splines.cubic_spline(
                inputs=inputs,
                unnormalized_widths=self.unnormalized_widths,
                unnormalized_heights=self.unnormalized_heights,
                unnorm_derivatives_left=self.unnorm_derivatives_left,
                unnorm_derivatives_right=self.unnorm_derivatives_right,
                inverse=inverse,
            )

        inputs = torch.rand(*self.shape)
        outputs, logabsdet = call_spline_fn(inputs, inverse=False)
        inputs_inv, logabsdet_inv = call_spline_fn(outputs, inverse=True)

        self.eps = 1e-3
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))

    def test_forward_inverse_are_consistent_in_tails(self):
        def call_spline_fn(inputs, inverse=False):
            return splines.unconstrained_cubic_spline(
                inputs=inputs,
                unnormalized_widths=self.unnormalized_widths,
                unnormalized_heights=self.unnormalized_heights,
                unnorm_derivatives_left=self.unnorm_derivatives_left,
                unnorm_derivatives_right=self.unnorm_derivatives_right,
                inverse=inverse,
                tail_bound=self.tail_bound
            )

        # Additional test for unconstrained cubic spline
        inputs = torch.sign(torch.randn(*self.shape)) * (self.tail_bound + torch.rand(*self.shape))
        outputs, logabsdet = call_spline_fn(inputs, inverse=False)
        inputs_inv, logabsdet_inv = call_spline_fn(outputs, inverse=True)

        self.eps = 1e-3
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))