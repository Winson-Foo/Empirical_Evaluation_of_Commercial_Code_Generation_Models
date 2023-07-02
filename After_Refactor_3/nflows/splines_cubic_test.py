import torch
import torchtestcase

from nflows.transforms import splines


class CubicSplineTest(torchtestcase.TorchTestCase):
    def test_forward_inverse_are_consistent(self):
        num_bins = 10
        shape = [2, 3, 4]

        unnormalized_widths = torch.randn(*shape, num_bins)
        unnormalized_heights = torch.randn(*shape, num_bins)
        unnorm_derivatives_left = torch.randn(*shape, 1)
        unnorm_derivatives_right = torch.randn(*shape, 1)

        inputs = torch.rand(*shape)
        outputs, logabsdet = self.call_spline_fn(
            splines.cubic_spline, inputs, unnormalized_widths, unnormalized_heights, unnorm_derivatives_left,
            unnorm_derivatives_right, inverse=False
        )
        inputs_inv, logabsdet_inv = self.call_spline_fn(
            splines.cubic_spline, outputs, unnormalized_widths, unnormalized_heights, unnorm_derivatives_left,
            unnorm_derivatives_right, inverse=True
        )

        self.eps = 1e-3
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))

    def call_spline_fn(self, spline_fn, inputs, unnormalized_widths, unnormalized_heights, unnorm_derivatives_left,
                       unnorm_derivatives_right, inverse=False):
        return spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnorm_derivatives_left=unnorm_derivatives_left,
            unnorm_derivatives_right=unnorm_derivatives_right,
            inverse=inverse,
        )


class UnconstrainedCubicSplineTest(torchtestcase.TorchTestCase):
    def test_forward_inverse_are_consistent(self):
        num_bins = 10
        shape = [2, 3, 4]
        tail_bound = 1.0

        unnormalized_widths = torch.randn(*shape, num_bins)
        unnormalized_heights = torch.randn(*shape, num_bins)
        unnorm_derivatives_left = torch.randn(*shape, 1)
        unnorm_derivatives_right = torch.randn(*shape, 1)

        inputs = 3 * torch.randn(*shape)
        outputs, logabsdet = self.call_spline_fn(
            splines.unconstrained_cubic_spline, inputs, unnormalized_widths, unnormalized_heights,
            unnorm_derivatives_left, unnorm_derivatives_right, inverse=False, tail_bound=tail_bound
        )
        inputs_inv, logabsdet_inv = self.call_spline_fn(
            splines.unconstrained_cubic_spline, outputs, unnormalized_widths, unnormalized_heights,
            unnorm_derivatives_left, unnorm_derivatives_right, inverse=True, tail_bound=tail_bound
        )

        self.eps = 1e-3
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))

    def test_forward_inverse_are_consistent_in_tails(self):
        num_bins = 10
        shape = [2, 3, 4]
        tail_bound = 1.0

        unnormalized_widths = torch.randn(*shape, num_bins)
        unnormalized_heights = torch.randn(*shape, num_bins)
        unnorm_derivatives_left = torch.randn(*shape, 1)
        unnorm_derivatives_right = torch.randn(*shape, 1)

        inputs = torch.sign(torch.randn(*shape)) * (tail_bound + torch.rand(*shape))
        outputs, logabsdet = self.call_spline_fn(
            splines.unconstrained_cubic_spline, inputs, unnormalized_widths, unnormalized_heights,
            unnorm_derivatives_left, unnorm_derivatives_right, inverse=False, tail_bound=tail_bound
        )
        inputs_inv, logabsdet_inv = self.call_spline_fn(
            splines.unconstrained_cubic_spline, outputs, unnormalized_widths, unnormalized_heights,
            unnorm_derivatives_left, unnorm_derivatives_right, inverse=True, tail_bound=tail_bound
        )

        self.eps = 1e-3
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))

    def call_spline_fn(self, spline_fn, inputs, unnormalized_widths, unnormalized_heights, unnorm_derivatives_left,
                       unnorm_derivatives_right, inverse=False, tail_bound=None):
        return spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnorm_derivatives_left=unnorm_derivatives_left,
            unnorm_derivatives_right=unnorm_derivatives_right,
            inverse=inverse,
            tail_bound=tail_bound
        )