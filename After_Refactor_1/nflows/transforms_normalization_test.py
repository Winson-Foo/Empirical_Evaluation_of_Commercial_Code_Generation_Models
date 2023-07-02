import unittest
import torch
from nflows.transforms import base
from nflows.transforms import normalization as norm
from tests.transforms.transform_test import TransformTest

class BatchNormTest(TransformTest):
    def _compute_outputs(self, inputs, transform):
        outputs, logabsdet = transform(inputs)
        self.assert_tensor_is_good(outputs, [inputs.size(0), inputs.size(1)])
        self.assert_tensor_is_good(logabsdet, [inputs.size(0)])

        mean, var = inputs.mean(0), inputs.var(0)
        outputs_ref = (inputs - mean) / torch.sqrt(var + bn_eps)
        logabsdet_ref = torch.sum(torch.log(1.0 / torch.sqrt(var + bn_eps)))
        logabsdet_ref = torch.full([inputs.size(0)], logabsdet_ref.item())
        if affine:
            outputs_ref *= transform.weight
            outputs_ref += transform.bias
            logabsdet_ref += torch.sum(torch.log(transform.weight))

        return outputs, outputs_ref, logabsdet, logabsdet_ref

    def test_forward(self):
        features = 100
        batch_size = 50
        bn_eps = 1e-5
        self.eps = 1e-4
        for affine in [True, True]:
            with self.subTest(affine=affine):
                inputs = torch.randn(batch_size, features)
                transform = norm.BatchNorm(features=features, affine=affine, eps=bn_eps)

                outputs, outputs_ref, logabsdet, logabsdet_ref = self._compute_outputs(inputs, transform)
                self.assertEqual(outputs, outputs_ref)
                self.assertEqual(logabsdet, logabsdet_ref)

                transform.eval()
                outputs, outputs_ref, logabsdet, logabsdet_ref = self._compute_outputs(inputs, transform)
                self.assertEqual(outputs, outputs_ref)
                self.assertEqual(logabsdet, logabsdet_ref)

    def test_inverse(self):
        features = 100
        batch_size = 50
        inputs = torch.randn(batch_size, features)
        for affine in [True, False]:
            with self.subTest(affine=affine):
                transform = norm.BatchNorm(features=features, affine=affine)
                with self.assertRaises(base.InverseNotAvailable):
                    transform.inverse(inputs)
                transform.eval()
                outputs, logabsdet = transform.inverse(inputs)
                self.assert_tensor_is_good(outputs, [batch_size, features])
                self.assert_tensor_is_good(logabsdet, [batch_size])

    def test_forward_inverse_are_consistent(self):
        features = 100
        batch_size = 50
        inputs = torch.randn(batch_size, features)
        transforms = [norm.BatchNorm(features=features, affine=affine) for affine in [True, False]]
        self.eps = 1e-6
        for transform in transforms:
            with self.subTest(transform=transform):
                transform.eval()
                self.assert_forward_inverse_are_consistent(transform, inputs)