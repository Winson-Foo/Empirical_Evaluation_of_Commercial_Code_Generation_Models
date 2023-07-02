import torch
import torchtestcase
from nflows.transforms import splines

class RationalQuadraticSplineTest(torchtestcase.TorchTestCase):
    def setUp(self):
        super().setUp()
        self.num_bins = 10
        self.shape = [2, 3, 4]
        self.unnormalized_widths = torch.randn(*self.shape, self.num_bins)
        self.unnormalized_heights = torch.randn(*self.shape, self.num_bins)
        self.unnormalized_derivatives = torch.randn(*self.shape, self.num_bins + 1)

    def call_spline_fn(self, inputs, inverse=False):
        return splines.rational_quadratic_spline(
            inputs=inputs,
            unnormalized_widths=self.unnormalized_widths,
            unnormalized_heights=self.unnormalized_heights,
            unnormalized_derivatives=self.unnormalized_derivatives,
            inverse=inverse,
        )

    def test_forward_inverse_are_consistent(self):
        inputs = torch.rand(*self.shape)
        outputs, logabsdet = self.call_spline_fn(inputs, inverse=False)
        inputs_inv, logabsdet_inv = self.call_spline_fn(outputs, inverse=True)

        self.eps = 1e-3
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))

    def test_identity_init(self):
        self.unnormalized_widths.fill_(0)
        self.unnormalized_heights.fill_(0)
        self.unnormalized_derivatives.fill_(0)

        inputs = torch.rand(*self.shape)
        outputs, logabsdet = self.call_spline_fn(inputs, inverse=False)

        self.eps = 1e-6
        self.assertEqual(inputs, outputs)
        self.assertEqual(logabsdet, torch.zeros_like(logabsdet))

        inputs = torch.rand(*self.shape)
        outputs, logabsdet = self.call_spline_fn(inputs, inverse=True)

        self.assertEqual(inputs, outputs)
        self.assertEqual(logabsdet, torch.zeros_like(logabsdet))

class UnconstrainedRationalQuadraticSplineTest(RationalQuadraticSplineTest):
    def test_forward_inverse_are_consistent_in_tails(self):
        tail_bound = 1.0

        inputs = torch.sign(torch.randn(*self.shape)) * (tail_bound + torch.rand(*self.shape))
        outputs, logabsdet = self.call_spline_fn(inputs, inverse=False)
        inputs_inv, logabsdet_inv = self.call_spline_fn(outputs, inverse=True)

        self.eps = 1e-3
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))

    def test_identity_init(self):
        tail_bound = 1.0

        self.unnormalized_widths.fill_(0)
        self.unnormalized_heights.fill_(0)
        self.unnormalized_derivatives.fill_(0)

        inputs = torch.sign(torch.randn(*self.shape)) * (tail_bound + torch.rand(*self.shape))
        outputs, logabsdet = self.call_spline_fn(inputs, inverse=False)

        self.eps = 1e-6
        self.assertEqual(inputs, outputs)
        self.assertEqual(logabsdet, torch.zeros_like(logabsdet))

        inputs = torch.rand(*self.shape)
        outputs, logabsdet = self.call_spline_fn(inputs, inverse=True)

        self.assertEqual(inputs, outputs)
        self.assertEqual(logabsdet, torch.zeros_like(logabsdet))