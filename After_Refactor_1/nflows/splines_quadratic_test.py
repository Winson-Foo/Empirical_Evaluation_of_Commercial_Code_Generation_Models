import torch
import torchtestcase

from nflows.transforms import splines


class SplineTestBase(torchtestcase.TorchTestCase):
    def setUp(self):
        super().setUp()
        self.num_bins = 10
        self.shape = [2, 3, 4]
        self.unnormalized_widths = torch.randn(*self.shape, self.num_bins)

    def call_spline_fn(self, inputs, inverse=False):
        raise NotImplementedError

    def test_forward_inverse_are_consistent(self):
        inputs = torch.rand(*self.shape)
        outputs, logabsdet = self.call_spline_fn(inputs, inverse=False)
        inputs_inv, logabsdet_inv = self.call_spline_fn(outputs, inverse=True)

        self.eps = 1e-3
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))


class QuadraticSplineTest(SplineTestBase):
    def setUp(self):
        super().setUp()
        self.unnormalized_heights = torch.randn(*self.shape, self.num_bins + 1)

    def call_spline_fn(self, inputs, inverse=False):
          return splines.quadratic_spline(
              inputs=inputs,
              unnormalized_widths=self.unnormalized_widths,
              unnormalized_heights=self.unnormalized_heights,
              inverse=inverse,
          )


class UnconstrainedQuadraticSplineTest(SplineTestBase):
    def setUp(self):
        super().setUp()
        self.unnormalized_heights = torch.randn(*self.shape, self.num_bins - 1)
        self.tail_bound = 1.0

    def call_spline_fn(self, inputs, inverse=False):
        return splines.unconstrained_quadratic_spline(
            inputs=inputs,
            unnormalized_widths=self.unnormalized_widths,
            unnormalized_heights=self.unnormalized_heights,
            inverse=inverse,
            tail_bound=self.tail_bound,
        )

    def test_forward_inverse_are_consistent(self):
        inputs = 3 * torch.randn(*self.shape)  # Note inputs are outside [0,1].
        super().test_forward_inverse_are_consistent()

    def test_forward_inverse_are_consistent_in_tails(self):
        inputs = torch.sign(torch.randn(*self.shape)) * (self.tail_bound + torch.rand(*self.shape))  # Now *all* inputs are outside [-tail_bound, tail_bound].
        super().test_forward_inverse_are_consistent()