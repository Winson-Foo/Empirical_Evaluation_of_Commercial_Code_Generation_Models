from torch import Tensor
from torchtestcase import TorchTestCase

def call_spline_fn(inputs: Tensor, inverse: bool, unnormalized_pdf: Tensor, tail_bound: float = None) -> Tuple[Tensor, Tensor]:
    if tail_bound is not None:
        spline_fn = splines.unconstrained_linear_spline
    else:
        spline_fn = splines.linear_spline
    return spline_fn(inputs=inputs, unnormalized_pdf=unnormalized_pdf, inverse=inverse, tail_bound=tail_bound)

class LinearSplineTest(TorchTestCase):
    unnormalized_pdf = torch.randn(2, 3, 4, 10)
    shape = [2, 3, 4]

    def test_forward_inverse_are_consistent(self):
        inputs = torch.rand(*self.shape)
        outputs, logabsdet = call_spline_fn(inputs, inverse=False, unnormalized_pdf=self.unnormalized_pdf)
        inputs_inv, logabsdet_inv = call_spline_fn(outputs, inverse=True, unnormalized_pdf=self.unnormalized_pdf)

        self.assertEqual(inputs, inputs_inv)
        self.assertClose(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet), atol=1e-3)

class UnconstrainedLinearSplineTest(TorchTestCase):
    unnormalized_pdf = torch.randn(2, 3, 4, 10)
    shape = [2, 3, 4]
    tail_bound = 1.0

    def test_forward_inverse_are_consistent(self):
        inputs = 3 * torch.randn(*self.shape)
        outputs, logabsdet = call_spline_fn(inputs, inverse=False, unnormalized_pdf=self.unnormalized_pdf)
        inputs_inv, logabsdet_inv = call_spline_fn(outputs, inverse=True, unnormalized_pdf=self.unnormalized_pdf)

        self.assertEqual(inputs, inputs_inv)
        self.assertClose(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet), atol=1e-3)

    def test_forward_inverse_are_consistent_in_tails(self):
        inputs = torch.sign(torch.randn(*self.shape)) * (self.tail_bound + torch.rand(*self.shape))
        outputs, logabsdet = call_spline_fn(inputs, inverse=False, unnormalized_pdf=self.unnormalized_pdf, tail_bound=self.tail_bound)
        inputs_inv, logabsdet_inv = call_spline_fn(outputs, inverse=True, unnormalized_pdf=self.unnormalized_pdf, tail_bound=self.tail_bound)
        
        self.assertEqual(inputs, inputs_inv)
        self.assertClose(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet), atol=1e-3)