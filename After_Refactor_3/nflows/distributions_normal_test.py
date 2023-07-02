"""
Tests for ConditionalDiagonalNormal distribution.
"""

import unittest
import torch
from conditional_diagonal_normal import ConditionalDiagonalNormal

class ConditionalDiagonalNormalTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 10
        self.input_shape = [2, 3, 4]
        self.context_shape = [2, 3, 8]
        self.dist = ConditionalDiagonalNormal(self.input_shape)

    def test_log_prob(self):
        inputs = torch.randn(self.batch_size, *self.input_shape)
        context = torch.randn(self.batch_size, *self.context_shape)
        log_prob = self.dist.log_prob(inputs, context=context)
        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual(log_prob.shape, torch.Size([self.batch_size]))
        self.assertFalse(torch.isnan(log_prob).any())
        self.assertFalse(torch.isinf(log_prob).any())

    def test_sample(self):
        num_samples = 10
        context_size = 20
        context = torch.randn(context_size, *self.context_shape)
        samples = self.dist.sample(num_samples, context=context)
        self.assertIsInstance(samples, torch.Tensor)
        self.assertEqual(
            samples.shape, torch.Size([context_size, num_samples] + self.input_shape)
        )
        self.assertFalse(torch.isnan(samples).any())
        self.assertFalse(torch.isinf(samples).any())

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
        context = torch.randn(context_size, *self.context_shape)
        means = self.dist.mean(context=context)
        self.assertIsInstance(means, torch.Tensor)
        self.assertFalse(torch.isnan(means).any())
        self.assertFalse(torch.isinf(means).any())
        self.assertEqual(means.shape, torch.Size([context_size] + self.input_shape))


if __name__ == "__main__":
    unittest.main()