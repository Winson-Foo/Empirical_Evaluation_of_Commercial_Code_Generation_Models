import torch
import torchtestcase
from nflows.transforms import splines


class SplineTestCase(torchtestcase.TorchTestCase):
    """
    Base test case class for spline transforms.
    """

    def _test_forward_inverse_consistency(self, spline_fn, num_bins, shape, tail_bound=None):
        """
        Helper function to test the forward and inverse consistency of a given spline function.
        """

        unnormalized_widths = torch.randn(*shape, num_bins)
        unnormalized_heights = torch.randn(*shape, num_bins - 1)

        def call_spline_fn(inputs, inverse=False):
            return spline_fn(
                inputs=inputs,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                inverse=inverse,
                tail_bound=tail_bound,
            )

        inputs = self._generate_input(shape, tail_bound)
        outputs, logabsdet = call_spline_fn(inputs, inverse=False)
        inputs_inv, logabsdet_inv = call_spline_fn(outputs, inverse=True)

        self.eps = 1e-3
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))

    def _generate_input(self, shape, tail_bound=None):
        """
        Helper function to generate input tensor for testing.
        """

        if tail_bound:
            inputs = torch.sign(torch.randn(*shape)) * (tail_bound + torch.rand(*shape))
        else:
            inputs = torch.rand(*shape)

        return inputs


class QuadraticSplineTest(SplineTestCase):
    """
    Class for testing the quadratic spline transform.
    """

    def test_forward_inverse_consistency(self):
        num_bins = 10
        shape = [2, 3, 4]

        self._test_forward_inverse_consistency(
            spline_fn=splines.quadratic_spline,
            num_bins=num_bins,
            shape=shape
        )


class UnconstrainedQuadraticSplineTest(SplineTestCase):
    """
    Class for testing the unconstrained quadratic spline transform.
    """

    def test_forward_inverse_consistency(self):
        num_bins = 10
        shape = [2, 3, 4]

        self._test_forward_inverse_consistency(
            spline_fn=splines.unconstrained_quadratic_spline,
            num_bins=num_bins,
            shape=shape
        )

    def test_forward_inverse_consistency_in_tails(self):
        num_bins = 10
        shape = [2, 3, 4]
        tail_bound = 1.0

        self._test_forward_inverse_consistency(
            spline_fn=splines.unconstrained_quadratic_spline,
            num_bins=num_bins,
            shape=shape,
            tail_bound=tail_bound
        )