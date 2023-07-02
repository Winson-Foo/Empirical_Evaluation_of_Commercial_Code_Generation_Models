import unittest
import torch
import torchtestcase
from nflows.distributions.normal import StandardNormal
from nflows.flows import base
from nflows.transforms.standard import AffineScalarTransform


class FlowTest(torchtestcase.TorchTestCase):
    def setUp(self):
        self.batch_size = 10
        self.num_samples = 10
        self.context_size = 20
        self.input_shape = [2, 3, 4]
        self.context_shape = [5, 6]
        self.flow = base.Flow(
            transform=AffineScalarTransform(scale=2.0),
            distribution=StandardNormal(self.input_shape),
        )

    def test_log_prob(self):
        inputs = torch.randn(self.batch_size, *self.input_shape)
        maybe_context = torch.randn(self.batch_size, *self.context_shape)
        for context in [None, maybe_context]:
            with self.subTest(context=context):
                log_prob = self.flow.log_prob(inputs, context=context)
                self.assertIsInstance(log_prob, torch.Tensor)
                self.assertEqual(log_prob.shape, torch.Size([self.batch_size]))

    def test_sample(self):
        maybe_context = torch.randn(self.context_size, *self.context_shape)
        for context in [None, maybe_context]:
            with self.subTest(context=context):
                samples = self.flow.sample(self.num_samples, context=context)
                self.assertIsInstance(samples, torch.Tensor)
                if context is None:
                    self.assertEqual(
                        samples.shape, torch.Size([self.num_samples] + self.input_shape)
                    )
                else:
                    self.assertEqual(
                        samples.shape,
                        torch.Size([self.context_size, self.num_samples] + self.input_shape),
                    )

    def test_sample_and_log_prob(self):
        samples, log_prob_1 = self.flow.sample_and_log_prob(self.num_samples)
        log_prob_2 = self.flow.log_prob(samples)
        self.assertIsInstance(samples, torch.Tensor)
        self.assertIsInstance(log_prob_1, torch.Tensor)
        self.assertIsInstance(log_prob_2, torch.Tensor)
        self.assertEqual(samples.shape, torch.Size([self.num_samples] + self.input_shape))
        self.assertEqual(log_prob_1.shape, torch.Size([self.num_samples]))
        self.assertEqual(log_prob_2.shape, torch.Size([self.num_samples]))
        self.assertEqual(log_prob_1, log_prob_2)

    def test_sample_and_log_prob_with_context(self):
        context = torch.randn(self.context_size, *self.context_shape)
        samples, log_prob = self.flow.sample_and_log_prob(self.num_samples, context=context)
        self.assertIsInstance(samples, torch.Tensor)
        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual(
            samples.shape, torch.Size([self.context_size, self.num_samples] + self.input_shape)
        )
        self.assertEqual(log_prob.shape, torch.Size([self.context_size, self.num_samples]))

    def test_transform_to_noise(self):
        inputs = torch.randn(self.batch_size, *self.input_shape)
        maybe_context = torch.randn(self.context_size, *self.context_shape)
        for context in [None, maybe_context]:
            with self.subTest(context=context):
                noise = self.flow.transform_to_noise(inputs, context=context)
                self.assertIsInstance(noise, torch.Tensor)
                self.assertEqual(noise.shape, torch.Size([self.batch_size] + self.input_shape))


if __name__ == "__main__":
    unittest.main()