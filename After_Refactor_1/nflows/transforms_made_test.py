import unittest
import torch
import torchtestcase
from nflows.transforms import made

class ShapeTest(torchtestcase.TorchTestCase):
  
    def test_conditional(self):
        self._test_made_model_with_inputs(
            features=100,
            hidden_features=200,
            num_blocks=5,
            output_multiplier=3,
            context_features=50,
            batch_size=16
        )
    
    def test_unconditional(self):
        self._test_made_model_with_inputs(
            features=100,
            hidden_features=200,
            num_blocks=5,
            output_multiplier=3,
            context_features=None,
            batch_size=16
        )
    
    def _test_made_model_with_inputs(
        self,
        features,
        hidden_features,
        num_blocks,
        output_multiplier,
        context_features,
        batch_size
    ):
        inputs = torch.randn(batch_size, features)
        conditional_inputs = (
            torch.randn(batch_size, context_features) 
            if context_features is not None else None
        )

        for use_residual_blocks, random_mask in [
            (False, False),
            (False, True),
            (True, False),
        ]:
            with self.subTest(
                use_residual_blocks=use_residual_blocks, random_mask=random_mask
            ):
                model = made.MADE(
                    features=features,
                    hidden_features=hidden_features,
                    num_blocks=num_blocks,
                    output_multiplier=output_multiplier,
                    context_features=context_features,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=random_mask,
                )
                outputs = model(inputs, conditional_inputs)
                self.assertEqual(outputs.dim(), 2)
                self.assertEqual(outputs.shape[0], batch_size)
                self.assertEqual(outputs.shape[1], output_multiplier * features)

class ConnectivityTest(torchtestcase.TorchTestCase):
  
    def test_gradients(self):
        self._test_made_gradients(
            features=10,
            hidden_features=256,
            num_blocks=20,
            output_multiplier=3
        )
  
    def test_total_mask_sequential(self):
        self._test_total_mask(use_residual_blocks=True)
        self._test_total_mask(use_residual_blocks=False)
    
    def test_total_mask_random(self):
        self._test_total_mask(use_residual_blocks=False, random_mask=True)
    
    def _test_made_gradients(
        self,
        features,
        hidden_features,
        num_blocks,
        output_multiplier
    ):
        for use_residual_blocks, random_mask in [
            (False, False),
            (False, True),
            (True, False),
        ]:
            with self.subTest(
                use_residual_blocks=use_residual_blocks, random_mask=random_mask
            ):
                model = made.MADE(
                    features=features,
                    hidden_features=hidden_features,
                    num_blocks=num_blocks,
                    output_multiplier=output_multiplier,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=random_mask,
                )
                inputs = torch.randn(1, features)
                inputs.requires_grad = True
                for k in range(features * output_multiplier):
                    outputs = model(inputs)
                    outputs[0, k].backward()
                    depends = inputs.grad.data[0] != 0.0
                    dim = k // output_multiplier
                    self.assertEqual(torch.all(depends[dim:] == 0), 1)
  
    def _test_total_mask(self, use_residual_blocks, random_mask=False):
        features = 10
        hidden_features = 50
        num_blocks = 5
        output_multiplier = 1

        model = made.MADE(
            features=features,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            output_multiplier=output_multiplier,
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
        )
        total_mask = model.initial_layer.mask
        for block in model.blocks:
            if use_residual_blocks:
                self.assertIsInstance(block, made.MaskedResidualBlock)
                total_mask = block.linear_layers[0].mask @ total_mask
                total_mask = block.linear_layers[1].mask @ total_mask
            else:
                self.assertIsInstance(block, made.MaskedFeedforwardBlock)
                total_mask = block.linear.mask @ total_mask
        total_mask = model.final_layer.mask @ total_mask
        total_mask = (total_mask > 0).float()
        reference = torch.tril(torch.ones([features, features]), -1)
        self.assertEqual(total_mask, reference)

if __name__ == "__main__":
    unittest.main()