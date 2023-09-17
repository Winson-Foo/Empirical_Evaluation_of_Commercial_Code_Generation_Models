def setup_spline_params(shape, num_bins):
    unnormalized_widths = torch.randn(*shape, num_bins)
    unnormalized_heights = torch.randn(*shape, num_bins)
    unnorm_derivatives_left = torch.randn(*shape, 1)
    unnorm_derivatives_right = torch.randn(*shape, 1)
    return unnormalized_widths, unnormalized_heights, unnorm_derivatives_left, unnorm_derivatives_right

def call_spline_fn(inputs, unnormalized_widths, unnormalized_heights, unnorm_derivatives_left, unnorm_derivatives_right, inverse=False, tail_bound=None):
    if tail_bound is None:
        return splines.cubic_spline(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnorm_derivatives_left=unnorm_derivatives_left,
            unnorm_derivatives_right=unnorm_derivatives_right,
            inverse=inverse,
        )
    else:
        return splines.unconstrained_cubic_spline(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnorm_derivatives_left=unnorm_derivatives_left,
            unnorm_derivatives_right=unnorm_derivatives_right,
            inverse=inverse,
            tail_bound=tail_bound
        )

class CubicSplineTest(torchtestcase.TorchTestCase):
    def test_forward_inverse_are_consistent(self):
        num_bins = 10
        shape = [2, 3, 4]

        unnormalized_widths, unnormalized_heights, unnorm_derivatives_left, unnorm_derivatives_right = setup_spline_params(shape, num_bins)

        inputs = torch.rand(*shape)
        outputs, logabsdet = call_spline_fn(inputs, unnormalized_widths, unnormalized_heights, unnorm_derivatives_left, unnorm_derivatives_right, inverse=False)
        inputs_inv, logabsdet_inv = call_spline_fn(outputs, unnormalized_widths, unnormalized_heights, unnorm_derivatives_left, unnorm_derivatives_right, inverse=True)

        self.eps = 1e-3
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))


class UnconstrainedCubicSplineTest(torchtestcase.TorchTestCase):
    def test_forward_inverse_are_consistent(self):
        num_bins = 10
        shape = [2, 3, 4]
        tail_bound = 1.0

        unnormalized_widths, unnormalized_heights, unnorm_derivatives_left, unnorm_derivatives_right = setup_spline_params(shape, num_bins)

        inputs = 3 * torch.randn(*shape)  # Note inputs are outside [0,1].
        outputs, logabsdet = call_spline_fn(inputs, unnormalized_widths, unnormalized_heights, unnorm_derivatives_left, unnorm_derivatives_right, inverse=False)
        inputs_inv, logabsdet_inv = call_spline_fn(outputs, unnormalized_widths, unnormalized_heights, unnorm_derivatives_left, unnorm_derivatives_right, inverse=True)

        self.eps = 1e-3
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))

    def test_forward_inverse_are_consistent_in_tails(self):
        num_bins = 10
        shape = [2, 3, 4]
        tail_bound = 1.0

        unnormalized_widths, unnormalized_heights, unnorm_derivatives_left, unnorm_derivatives_right = setup_spline_params(shape, num_bins)

        inputs = torch.sign(torch.randn(*shape)) * (tail_bound + torch.rand(*shape))  # Now *all* inputs are outside [-tail_bound, tail_bound].
        outputs, logabsdet = call_spline_fn(inputs, unnormalized_widths, unnormalized_heights, unnorm_derivatives_left, unnorm_derivatives_right, inverse=False)
        inputs_inv, logabsdet_inv = call_spline_fn(outputs, unnormalized_widths, unnormalized_heights, unnorm_derivatives_left, unnorm_derivatives_right, inverse=True)

        self.eps = 1e-3
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))