import unittest

import torch
import torchtestcase
from nflows.distributions import discrete


class ConditionalIndependentBernoulliTest(torchtestcase.TorchTestCase):
    def setUp(self):
        self.batch_size = 10
        self.input_shape = [2, 3, 4]
        self.context_shape = [2, 3, 4]
        self.num_samples = 10
        self.context_size = 20

        self.dist = discrete.ConditionalIndependentBernoulli(self.input_shape)

        self.inputs = torch.randn(self.batch_size, *self.input_shape)
        self.context = torch.randn(self.batch_size, *self.context_shape)
        self.context_batch = torch.randn(self.context_size, *self.context_shape)

    def test_log_prob(self):
        log_prob = self.dist.log_prob(self.inputs, context=self.context)

        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual(log_prob.shape, torch.Size([self.batch_size]))
        self.assertFalse(torch.isnan(log_prob).any())
        self.assertFalse(torch.isinf(log_prob).any())
        self.assert_tensor_less_equal(log_prob, 0.0)

    def test_sample(self):
        samples = self.dist.sample(self.num_samples, context=self.context_batch)

        self.assertIsInstance(samples, torch.Tensor)
        self.assertEqual(
            samples.shape, torch.Size([self.context_size, self.num_samples] + self.input_shape)
        )
        self.assertFalse(torch.isnan(samples).any())
        self.assertFalse(torch.isinf(samples).any())
        binary = (samples == 1.0) | (samples == 0.0)
        self.assertEqual(binary, torch.ones_like(binary))

    def test_sample_and_log_prob_with_context(self):
        samples, log_prob = self.dist.sample_and_log_prob(self.num_samples, context=self.context_batch)

        self.assertIsInstance(samples, torch.Tensor)
        self.assertIsInstance(log_prob, torch.Tensor)

        self.assertEqual(
            samples.shape, torch.Size([self.context_size, self.num_samples] + self.input_shape)
        )
        self.assertEqual(log_prob.shape, torch.Size([self.context_size, self.num_samples]))

        self.assertFalse(torch.isnan(log_prob).any())
        self.assertFalse(torch.isinf(log_prob).any())
        self.assert_tensor_less_equal(log_prob, 0.0)

        self.assertFalse(torch.isnan(samples).any())
        self.assertFalse(torch.isinf(samples).any())
        binary = (samples == 1.0) | (samples == 0.0)
        self.assertEqual(binary, torch.ones_like(binary))

    def test_mean(self):
        means = self.dist.mean(context=self.context_batch)

        self.assertIsInstance(means, torch.Tensor)
        self.assertEqual(means.shape, torch.Size([self.context_size] + self.input_shape))
        self.assertFalse(torch.isnan(means).any())
        self.assertFalse(torch.isinf(means).any())
        self.assert_tensor_greater_equal(means, 0.0)
        self.assert_tensor_less_equal(means, 1.0)


if __name__ == "__main__":
    unittest.main()