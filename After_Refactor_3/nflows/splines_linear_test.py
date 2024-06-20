import torch
import torchtestcase

from nflows.transforms import splines


class SplineTest(torchtestcase.TorchTestCase):
    def call_spline_fn(self, inputs, unnormalized_pdf, inverse=False, tail_bound=None):
        if tail_bound is None:
            return splines.linear_spline(inputs=inputs, unnormalized_pdf=unnormalized_pdf, inverse=inverse)
        else:
            return splines.unconstrained_linear_spline(inputs=inputs, unnormalized_pdf=unnormalized_pdf, inverse=inverse, tail_bound=tail_bound)

    def test_linear_spline_forward_inverse_are_consistent(self):
        num_bins = 10
        shape = [2, 3, 4]
    
        unnormalized_pdf = torch.randn(*shape, num_bins)
    
        inputs = torch.rand(*shape)
        outputs, logabsdet = self.call_spline_fn(inputs, unnormalized_pdf, inverse=False)
        inputs_inv, logabsdet_inv = self.call_spline_fn(outputs, unnormalized_pdf, inverse=True)
    
        self.eps = 1e-3
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))
    
    def test_unconstrained_linear_spline_forward_inverse_are_consistent(self):
        num_bins = 10
        shape = [2, 3, 4]
    
        unnormalized_pdf = torch.randn(*shape, num_bins)
    
        inputs = 3 * torch.randn(*shape)  # Note inputs are outside [0,1].
        outputs, logabsdet = self.call_spline_fn(inputs, unnormalized_pdf, inverse=False)
        inputs_inv, logabsdet_inv = self.call_spline_fn(outputs, unnormalized_pdf, inverse=True)
    
        self.eps = 1e-3
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))
    
    def test_unconstrained_linear_spline_forward_inverse_are_consistent_in_tails(self):
        num_bins = 10
        shape = [2, 3, 4]
        tail_bound = 1.0
    
        unnormalized_pdf = torch.randn(*shape, num_bins)
    
        inputs = torch.sign(torch.randn(*shape)) * (tail_bound + torch.rand(*shape))  # Now *all* inputs are outside [-tail_bound, tail_bound].
        outputs, logabsdet = self.call_spline_fn(inputs, unnormalized_pdf, inverse=False, tail_bound=tail_bound)
        inputs_inv, logabsdet_inv = self.call_spline_fn(outputs, unnormalized_pdf, inverse=True, tail_bound=tail_bound)
    
        self.eps = 1e-3
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))