import unittest

import torch
import torchtestcase

from nflows.flows import autoregressive as ar


class MaskedAutoregressiveFlowTest(torchtestcase.TorchTestCase):
    def test_log_prob(self):
        batch_size = 10
        num_features = 20
        hidden_features = 30
        num_layers = 5
        num_blocks_per_layer = 2
        
        flow = ar.MaskedAutoregressiveFlow(
            features=num_features, hidden_features=hidden_features,
            num_layers=num_layers, num_blocks_per_layer=num_blocks_per_layer
        )
        
        inputs = torch.randn(batch_size, num_features)
        log_prob = flow.log_prob(inputs)
        
        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual(log_prob.shape, torch.Size([batch_size]))
    
    def test_sample(self):
        num_samples = 10
        num_features = 20
        hidden_features = 30
        num_layers = 5
        num_blocks_per_layer = 2
        
        flow = ar.MaskedAutoregressiveFlow(
            features=num_features, hidden_features=hidden_features,
            num_layers=num_layers, num_blocks_per_layer=num_blocks_per_layer
        )
        
        samples = flow.sample(num_samples)
        
        self.assertIsInstance(samples, torch.Tensor)
        self.assertEqual(samples.shape, torch.Size([num_samples, num_features]))


if __name__ == "__main__":
    unittest.main()