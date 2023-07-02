import torch
import torchtestcase
from nflows.transforms import splines


class SplineTest(torchtestcase.TorchTestCase):
    def assertForwardInverseConsistency(self, inputs, spline_fn):
        outputs, logabsdet = spline_fn(inputs, inverse=False)
        inputs_inv, logabsdet_inv = spline_fn(outputs, inverse=True)

        self.eps = 1e-3
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))

    def test_quadratic_spline_forward_inverse(self):
        num_bins = 10
        shape = [2, 3, 4]

        unnormalized_widths = torch.randn(*shape, num_bins)
        unnormalized_heights = torch.randn(*shape, num_bins + 1)

        def quadratic_spline(inputs, inverse=False):
            return splines.quadratic_spline(
                inputs=inputs,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                inverse=inverse,
            )

        inputs = torch.rand(*shape)
        self.assertForwardInverseConsistency(inputs, quadratic_spline)

    def test_unconstrained_quadratic_spline_forward_inverse(self):
        num_bins = 10
        shape = [2, 3, 4]

        unnormalized_widths = torch.randn(*shape, num_bins)
        unnormalized_heights = torch.randn(*shape, num_bins - 1)

        def unconstrained_quadratic_spline(inputs, inverse=False):
            return splines.unconstrained_quadratic_spline(
                inputs=inputs,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                inverse=inverse,
            )

        inputs = 3 * torch.randn(*shape)
        self.assertForwardInverseConsistency(inputs, unconstrained_quadratic_spline)

    def test_unconstrained_quadratic_spline_forward_inverse_in_tails(self):
        num_bins = 10
        shape = [2, 3, 4]
        tail_bound = 1.0

        unnormalized_widths = torch.randn(*shape, num_bins)
        unnormalized_heights = torch.randn(*shape, num_bins - 1)

        def unconstrained_quadratic_spline(inputs, inverse=False):
            return splines.unconstrained_quadratic_spline(
                inputs=inputs,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                inverse=inverse,
                tail_bound=tail_bound,
            )

        inputs = torch.sign(torch.randn(*shape)) * (tail_bound + torch.rand(*shape))
        self.assertForwardInverseConsistency(inputs, unconstrained_quadratic_spline)