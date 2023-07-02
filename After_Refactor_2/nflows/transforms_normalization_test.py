import unittest
import torch
from nflows.transforms import base
from nflows.transforms import normalization as norm
from tests.transforms.transform_test import TransformTest

class BatchNormTest(TransformTest):
    def test_forward(self):
        num_features = 100
        batch_size = 50
        batch_norm_epsilon = 1e-5
        self.eps = 1e-4

        for affine in [True, True]:
            with self.subTest(affine=affine):
                inputs = torch.randn(batch_size, num_features)
                transform = norm.BatchNorm(num_features, affine=affine, eps=batch_norm_epsilon)

                outputs, logabsdet = transform(inputs)

                self.assert_tensor_is_good(outputs, [batch_size, num_features])
                self.assert_tensor_is_good(logabsdet, [batch_size])

                mean, var = inputs.mean(0), inputs.var(0)
                outputs_ref = (inputs - mean) / torch.sqrt(var + batch_norm_epsilon)
                logabsdet_ref = torch.sum(torch.log(1.0 / torch.sqrt(var + batch_norm_epsilon)))
                logabsdet_ref = torch.full([batch_size], logabsdet_ref.item())
                
                if affine:
                    outputs_ref *= transform.weight
                    outputs_ref += transform.bias
                    logabsdet_ref += torch.sum(torch.log(transform.weight))
                
                self.assert_tensor_is_good(outputs_ref, [batch_size, num_features])
                self.assert_tensor_is_good(logabsdet_ref, [batch_size])

                self.assertEqual(outputs, outputs_ref)
                self.assertEqual(logabsdet, logabsdet_ref)

                transform.eval()
                outputs, logabsdet = transform(inputs)

                self.assert_tensor_is_good(outputs, [batch_size, num_features])
                self.assert_tensor_is_good(logabsdet, [batch_size])

                mean = transform.running_mean
                var = transform.running_var
                outputs_ref = (inputs - mean) / torch.sqrt(var + batch_norm_epsilon)
                logabsdet_ref = torch.sum(torch.log(1.0 / torch.sqrt(var + batch_norm_epsilon)))
                logabsdet_ref = torch.full([batch_size], logabsdet_ref.item())

                if affine:
                    outputs_ref *= transform.weight
                    outputs_ref += transform.bias
                    logabsdet_ref += torch.sum(torch.log(transform.weight))
                
                self.assert_tensor_is_good(outputs_ref, [batch_size, num_features])
                self.assert_tensor_is_good(logabsdet_ref, [batch_size])

                self.assertEqual(outputs, outputs_ref)
                self.assertEqual(logabsdet, logabsdet_ref)

    def test_inverse(self):
        num_features = 100
        batch_size = 50
        inputs = torch.randn(batch_size, num_features)

        for affine in [True, False]:
            with self.subTest(affine=affine):
                transform = norm.BatchNorm(features=num_features, affine=affine)
                with self.assertRaises(base.InverseNotAvailable):
                    transform.inverse(inputs)
                transform.eval()
                outputs, logabsdet = transform.inverse(inputs)
                self.assert_tensor_is_good(outputs, [batch_size, num_features])
                self.assert_tensor_is_good(logabsdet, [batch_size])

    def test_forward_inverse_are_consistent(self):
        num_features = 100
        batch_size = 50
        inputs = torch.randn(batch_size, num_features)
        transforms = [norm.BatchNorm(features=num_features, affine=affine) for affine in [True, False]]
        self.eps = 1e-6
        for transform in transforms:
            with self.subTest(transform=transform):
                transform.eval()
                self.assert_forward_inverse_are_consistent(transform, inputs)

class ActNormTest(TransformTest):
    def test_forward(self):
        batch_size = 50
        for shape in [(100,), (32, 8, 8)]:
            with self.subTest(shape=shape):
                inputs = torch.randn(batch_size, *shape)
                transform = norm.ActNorm(shape[0])

                outputs, logabsdet = transform.forward(inputs)
                self.assert_tensor_is_good(outputs, [batch_size] + list(shape))
                self.assert_tensor_is_good(logabsdet, [batch_size])

    def test_inverse(self):
        batch_size = 50
        for shape in [(100,), (32, 8, 8)]:
            with self.subTest(shape=shape):
                inputs = torch.randn(batch_size, *shape)
                transform = norm.ActNorm(shape[0])

                outputs, logabsdet = transform.inverse(inputs)
                self.assert_tensor_is_good(outputs, [batch_size] + list(shape))
                self.assert_tensor_is_good(logabsdet, [batch_size])

    def test_forward_inverse_are_consistent(self):
        batch_size = 50
        for shape in [(100,), (32, 8, 8)]:
            with self.subTest(shape=shape):
                inputs = torch.randn(batch_size, *shape)
                transform = norm.ActNorm(shape[0])
                transform.forward(inputs)
                self.eps = 1e-6
                self.assert_forward_inverse_are_consistent(transform, inputs)

    def test_save_load(self):
        batch_size = 50
        for shape in [(100,), (32, 8, 8)]:
            with self.subTest(shape=shape):
                inputs = torch.randn(batch_size, *shape)
                transform1 = norm.ActNorm(shape[0])
                outputs1, logabsdet1 = transform1.forward(inputs)
                state_dict = transform1.state_dict()
                transform2 = norm.ActNorm(shape[0])
                transform2.load_state_dict(state_dict)
                noise = torch.randn(batch_size, *shape)
                _ = transform2.forward(noise)
                outputs2, logabsdet2 = transform2.forward(inputs)
                self.eps = 1e-6
                self.assertEqual(transform1.log_scale, transform2.log_scale)
                self.assertEqual(transform1.shift, transform2.shift)
                self.assertEqual(outputs1, outputs2)
                self.assertEqual(logabsdet1, logabsdet2)

if __name__ == "__main__":
    unittest.main()