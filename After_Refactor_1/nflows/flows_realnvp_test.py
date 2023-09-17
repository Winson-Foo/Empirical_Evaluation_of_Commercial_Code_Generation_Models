import unittest
import torch
import torchtestcase
from nflows.flows import realnvp

class RealNVPTests(torchtestcase.TorchTestCase):
    def setUp(self):
        self.batch_size = 10
        self.num_samples = 10
        self.features = 20
        self.hidden_features = 30
        self.num_layers = 5
        self.num_blocks_per_layer = 2

    def _initialize_flow(self):
        flow = realnvp.SimpleRealNVP(
            features=self.features,
            hidden_features=self.hidden_features,
            num_layers=self.num_layers,
            num_blocks_per_layer=self.num_blocks_per_layer
        )
        return flow

    def test_log_prob(self):
        flow = self._initialize_flow()
        inputs = torch.randn(self.batch_size, self.features)
        log_prob = flow.log_prob(inputs)
        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual(log_prob.shape, torch.Size([self.batch_size]))

    def test_sample(self):
        flow = self._initialize_flow()
        samples = flow.sample(self.num_samples)
        self.assertIsInstance(samples, torch.Tensor)
        self.assertEqual(samples.shape, torch.Size([self.num_samples, self.features]))


if __name__ == "__main__":
    unittest.main()