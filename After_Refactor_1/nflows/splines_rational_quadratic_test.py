import torch
import torchtestcase

from nflows.transforms import splines


class SplineTest(torchtestcase.TorchTestCase):
    def setUp(self):
        self.num_bins = 10
        self.shape = [2, 3, 4]
        self.unnormalized_widths = torch.randn(*self.shape, self.num_bins)
        self.unnormalized_heights = torch.randn(*self.shape, self.num_bins)
        self.unnormalized_derivatives = torch.randn(*self.shape, self.num_bins + 1)
        self.tail_bound = 1.0

    def call_spline_fn(self, inputs, spline_fn, inverse=False):
        return spline_fn(
            inputs=inputs,
            unnormalized_widths=self.unnormalized_widths,
            unnormalized_heights=self.unnormalized_heights,
            unnormalized_derivatives=self.unnormalized_derivatives,
            inverse=inverse,
        )


class RationalQuadraticSplineTest(SplineTest):
    def test_forward_inverse_are_consistent(self):
        shape = self.shape

        inputs = torch.rand(*shape)
        outputs, logabsdet = self.call_spline_fn(inputs, splines.rational_quadratic_spline, inverse=False)
        inputs_inv, logabsdet_inv = self.call_spline_fn(outputs, splines.rational_quadratic_spline, inverse=True)

        self.eps = 1e-3
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))

    def test_identity_init(self):
        shape = self.shape

        inputs = torch.rand(*shape)
        outputs, logabsdet = self.call_spline_fn(inputs, splines.rational_quadratic_spline, inverse=False)

        self.eps = 1e-6
        self.assertEqual(inputs, outputs)
        self.assertEqual(logabsdet, torch.zeros_like(logabsdet))

        inputs = torch.rand(*shape)
        outputs, logabsdet = self.call_spline_fn(inputs, splines.rational_quadratic_spline, inverse=True)

        self.assertEqual(inputs, outputs)
        self.assertEqual(logabsdet, torch.zeros_like(logabsdet))


class UnconstrainedRationalQuadraticSplineTest(SplineTest):
    def test_forward_inverse_are_consistent(self):
        shape = self.shape

        inputs = 3 * torch.randn(*shape)  # Note inputs are outside [0,1].
        outputs, logabsdet = self.call_spline_fn(inputs, splines.unconstrained_rational_quadratic_spline, inverse=False)
        inputs_inv, logabsdet_inv = self.call_spline_fn(outputs, splines.unconstrained_rational_quadratic_spline, inverse=True)

        self.eps = 1e-3
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))

    def test_forward_inverse_are_consistent_in_tails(self):
        shape = self.shape

        inputs = torch.sign(torch.randn(*shape)) * (self.tail_bound + torch.rand(*shape))  # Now *all* inputs are outside [-tail_bound, tail_bound].
        outputs, logabsdet = self.call_spline_fn(inputs, splines.unconstrained_rational_quadratic_spline, inverse=False)
        inputs_inv, logabsdet_inv = self.call_spline_fn(outputs, splines.unconstrained_rational_quadratic_spline, inverse=True)

        self.eps = 1e-3
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))

    def test_identity_init(self):
        shape = self.shape

        inputs = torch.sign(torch.randn(*shape)) * (self.tail_bound + torch.rand(*shape))  # Now *all* inputs are outside [-tail_bound, tail_bound].
        outputs, logabsdet = self.call_spline_fn(inputs, splines.unconstrained_rational_quadratic_spline, inverse=False)

        self.eps = 1e-6
        self.assertEqual(inputs, outputs)
        self.assertEqual(logabsdet, torch.zeros_like(logabsdet))

        inputs = torch.rand(*shape)
        outputs, logabsdet = self.call_spline_fn(inputs, splines.unconstrained_rational_quadratic_spline, inverse=True)

        self.assertEqual(inputs, outputs)
        self.assertEqual(logabsdet, torch.zeros_like(logabsdet))