"""
Tests for StandardNormal distribution.
"""

import unittest
import torch
from standard_normal import StandardNormal

class StandardNormalTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 10
        self.input_shape = [2, 3, 4]
        self.context_shape = [5, 6]
        self.dist = StandardNormal(self.input_shape)

    def test_log_prob(self):
        inputs = torch.randn(self.batch_size, *self.input_shape)
        maybe_context = torch.randn(self.batch_size, *self.context_shape)
        for context in [None, maybe_context]:
            with self.subTest(context=context):
                log_prob = self.dist.log_prob(inputs, context=context)
                self.assertIsInstance(log_prob, torch.Tensor)
                self.assertEqual(log_prob.shape, torch.Size([self.batch_size]))
                self.assertFalse(torch.isnan(log_prob).any())
                self.assertFalse(torch.isinf(log_prob).any())

    def test_sample(self):
        num_samples = 10
        context_size = 20
        maybe_context = torch.randn(context_size, *self.context_shape)
        for context in [None, maybe_context]:
            with self.subTest(context=context):
                samples = self.dist.sample(num_samples, context=context)
                self.assertIsInstance(samples, torch.Tensor)
                self.assertFalse(torch.isnan(samples).any())
                self.assertFalse(torch.isinf(samples).any())
                if context is None:
                    self.assertEqual(
                        samples.shape, torch.Size([num_samples] + self.input_shape)
                    )
                else:
                    self.assertEqual(
                        samples.shape,
                        torch.Size([context_size, num_samples] + self.input_shape),
                    )

    def test_sample_and_log_prob(self):
        num_samples = 10
        inputs, log_prob_1 = self.dist.sample_and_log_prob(num_samples)
        log_prob_2 = self.dist.log_prob(inputs)
        self.assertIsInstance(inputs, torch.Tensor)
        self.assertIsInstance(log_prob_1, torch.Tensor)
        self.assertIsInstance(log_prob_2, torch.Tensor)
        self.assertEqual(inputs.shape, torch.Size([num_samples] + self.input_shape))
        self.assertEqual(log_prob_1.shape, torch.Size([num_samples]))
        self.assertEqual(log_prob_2.shape, torch.Size([num_samples]))
        self.assertEqual(log_prob_1, log_prob_2)

    def test_sample_and_log_prob_with_context(self):
        num_samples = 10
        context_size = 20
        context = torch.randn(context_size, *self.context_shape)
        samples, log_prob = self.dist.sample_and_log_prob(num_samples, context=context)
        self.assertIsInstance(samples, torch.Tensor)
        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual(
            samples.shape, torch.Size([context_size, num_samples] + self.input_shape)
        )
        self.assertEqual(log_prob.shape, torch.Size([context_size, num_samples]))

    def test_mean(self):
        context_size = 20
        maybe_context = torch.randn(context_size, *self.context_shape)
        for context in [None, maybe_context]:
            with self.subTest(context=context):
                means = self.dist.mean(context=context)
                self.assertIsInstance(means, torch.Tensor)
                self.assertFalse(torch.isnan(means).any())
                self.assertFalse(torch.isinf(means).any())
                self.assertEqual(means, torch.zeros_like(means))
                if context is None:
                    self.assertEqual(means.shape, torch.Size(self.input_shape))
                else:
                    self.assertEqual(
                        means.shape, torch.Size([context_size] + self.input_shape)
                    )

if __name__ == "__main__":
    unittest.main()