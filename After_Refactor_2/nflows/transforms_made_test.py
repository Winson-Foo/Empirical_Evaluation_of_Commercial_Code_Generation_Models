import unittest
import torch
import torchtestcase
from nflows.transforms import made

class ShapeTest(torchtestcase.TorchTestCase):
    def setUp(self):
        self.features = 100
        self.hidden_features = 200
        self.num_blocks = 5
        self.output_multiplier = 3
        self.context_features = 50
        self.batch_size = 16
        self.model = None

    def test_conditional(self):
        inputs = torch.randn(self.batch_size, self.features)
        conditional_inputs = torch.randn(self.batch_size, self.context_features)
        
        for use_residual_blocks, random_mask in [(False, False), (False, True), (True, False)]:
            with self.subTest(use_residual_blocks=use_residual_blocks, random_mask=random_mask):
                self._initialize_model(self.features, self.hidden_features, self.num_blocks, 
                                       self.output_multiplier, self.context_features, use_residual_blocks, random_mask)
                outputs = self.model(inputs, conditional_inputs)
                self.assertEqual(outputs.dim(), 2)
                self.assertEqual(outputs.shape[0], self.batch_size)
                self.assertEqual(outputs.shape[1], self.output_multiplier * self.features)

    def test_unconditional(self):
        inputs = torch.randn(self.batch_size, self.features)
        
        for use_residual_blocks, random_mask in [(False, False), (False, True), (True, False)]:
            with self.subTest(use_residual_blocks=use_residual_blocks, random_mask=random_mask):
                self._initialize_model(self.features, self.hidden_features, self.num_blocks, 
                                       self.output_multiplier, None, use_residual_blocks, random_mask)
                outputs = self.model(inputs)
                self.assertEqual(outputs.dim(), 2)
                self.assertEqual(outputs.shape[0], self.batch_size)
                self.assertEqual(outputs.shape[1], self.output_multiplier * self.features)
    
    def _initialize_model(self, features, hidden_features, num_blocks, 
                          output_multiplier, context_features, use_residual_blocks, random_mask):
        self.model = made.MADE(
            features=features,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            output_multiplier=output_multiplier,
            context_features=context_features,
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
        )


class ConnectivityTest(torchtestcase.TorchTestCase):
    def setUp(self):
        self.features = 10
        self.hidden_features = 256
        self.num_blocks = 20
        self.output_multiplier = 3
        self.model = None

    def test_gradients(self):
        inputs = torch.randn(1, self.features)
        inputs.requires_grad = True
        
        for use_residual_blocks, random_mask in [(False, False), (False, True), (True, False)]:
            with self.subTest(use_residual_blocks=use_residual_blocks, random_mask=random_mask):
                self._initialize_model(self.features, self.hidden_features, self.num_blocks, 
                                       self.output_multiplier, None, use_residual_blocks, random_mask)
                for k in range(self.features * self.output_multiplier):
                    outputs = self.model(inputs)
                    outputs[0, k].backward()
                    depends = inputs.grad.data[0] != 0.0
                    dim = k // self.output_multiplier
                    self.assertEqual(torch.all(depends[dim:] == 0), 1)

    def test_total_mask_sequential(self):
        for use_residual_blocks in [True, False]:
            with self.subTest(use_residual_blocks=use_residual_blocks):
                self._initialize_model(self.features, self.hidden_features, self.num_blocks, 
                                       self.output_multiplier, None, use_residual_blocks, False)
                total_mask = self.model.initial_layer.mask
                for block in self.model.blocks:
                    if use_residual_blocks:
                        self.assertIsInstance(block, made.MaskedResidualBlock)
                        total_mask = block.linear_layers[0].mask @ total_mask
                        total_mask = block.linear_layers[1].mask @ total_mask
                    else:
                        self.assertIsInstance(block, made.MaskedFeedforwardBlock)
                        total_mask = block.linear.mask @ total_mask
                total_mask = self.model.final_layer.mask @ total_mask
                total_mask = (total_mask > 0).float()
                reference = torch.tril(torch.ones([self.features, self.features]), -1)
                self.assertEqual(total_mask, reference)

    def test_total_mask_random(self):
        self._initialize_model(self.features, self.hidden_features, self.num_blocks, 
                               self.output_multiplier, None, False, True)
        total_mask = self.model.initial_layer.mask
        for block in self.model.blocks:
            self.assertIsInstance(block, made.MaskedFeedforwardBlock)
            total_mask = block.linear.mask @ total_mask
        total_mask = self.model.final_layer.mask @ total_mask
        total_mask = (total_mask > 0).float()
        self.assertEqual(torch.triu(total_mask), torch.zeros([self.features, self.features]))
    
    def _initialize_model(self, features, hidden_features, num_blocks, 
                          output_multiplier, context_features, use_residual_blocks, random_mask):
        self.model = made.MADE(
            features=features,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            output_multiplier=output_multiplier,
            context_features=context_features,
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
        )


if __name__ == "__main__":
    unittest.main()