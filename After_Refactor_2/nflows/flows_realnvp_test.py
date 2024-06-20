"""Tests for Real NVP."""

import unittest

import torch
import torchtestcase

from nflows.flows import realnvp


class SimpleRealNVPTest(torchtestcase.TorchTestCase):
    def setUp(self):
        self.features = 20
        self.hidden_features = 30
        self.num_layers = 5
        self.num_blocks_per_layer = 2

        self.flow = self._create_flow()

    def _create_flow(self):
        return realnvp.SimpleRealNVP(
            features=self.features,
            hidden_features=self.hidden_features,
            num_layers=self.num_layers,
            num_blocks_per_layer=self.num_blocks_per_layer,
        )

    def test_log_prob(self):
        batch_size = 10
        inputs = torch.randn(batch_size, self.features)

        log_prob = self.flow.log_prob(inputs)

        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual(log_prob.shape, torch.Size([batch_size]))

    def test_sample(self):
        num_samples = 10
        samples = self.flow.sample(num_samples)

        self.assertIsInstance(samples, torch.Tensor)
        self.assertEqual(samples.shape, torch.Size([num_samples, self.features]))


if __name__ == "__main__":
    unittest.main()