import unittest
import torch
import torchtestcase
from nflows.transforms import made


class MADETestCase(torchtestcase.TorchTestCase):
    def run_test(self, features, hidden_features, num_blocks, output_multiplier,
                 conditional_inputs=None, random_mask=False, use_residual_blocks=False):
        batch_size = 16
        inputs = torch.randn(batch_size, features)

        if conditional_inputs is not None:
            conditional_inputs = torch.randn(batch_size, conditional_inputs)

        model = made.MADE(
            features=features,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            output_multiplier=output_multiplier,
            context_features=conditional_inputs.shape[1] if conditional_inputs is not None else None,
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
        )

        if conditional_inputs is not None:
            outputs = model(inputs, conditional_inputs)
        else:
            outputs = model(inputs)

        self.assertEqual(outputs.dim(), 2)
        self.assertEqual(outputs.shape[0], batch_size)
        self.assertEqual(outputs.shape[1], output_multiplier * features)

    def test_conditional(self):
        self.run_test(features=100, hidden_features=200, num_blocks=5,
                      output_multiplier=3, conditional_inputs=50)

    def test_unconditional(self):
        self.run_test(features=100, hidden_features=200, num_blocks=5,
                      output_multiplier=3)

    def run_connectivity_test(self, features, hidden_features, num_blocks, output_multiplier,
                              use_residual_blocks=False, random_mask=False):
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

        return total_mask

    def test_gradients(self):
        features = 10
        hidden_features = 256
        num_blocks = 20
        output_multiplier = 3

        inputs = torch.randn(1, features)
        inputs.requires_grad = True
        for use_residual_blocks, random_mask in [(False, False), (False, True), (True, False)]:
            with self.subTest(use_residual_blocks=use_residual_blocks, random_mask=random_mask):
                model = made.MADE(
                    features=features,
                    hidden_features=hidden_features,
                    num_blocks=num_blocks,
                    output_multiplier=output_multiplier,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=random_mask,
                )
                outputs = model(inputs)
                outputs[0].backward()
                depends = inputs.grad.data[0] != 0.0
                dim = range(0, features * output_multiplier, output_multiplier)
                self.assertTrue(torch.all(depends[dim:] == 0))
    
    def test_total_mask_sequential(self):
        features = 10
        hidden_features = 50
        num_blocks = 5
        output_multiplier = 1

        for use_residual_blocks in [True, False]:
            with self.subTest(use_residual_blocks=use_residual_blocks):
                total_mask = self.run_connectivity_test(
                    features=features,
                    hidden_features=hidden_features,
                    num_blocks=num_blocks,
                    output_multiplier=output_multiplier,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=False,
                )
                reference = torch.tril(torch.ones([features, features]), -1)
                self.assertEqual(total_mask, reference)

    def test_total_mask_random(self):
        features = 10
        hidden_features = 50
        num_blocks = 5
        output_multiplier = 1

        total_mask = self.run_connectivity_test(
            features=features,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            output_multiplier=output_multiplier,
            use_residual_blocks=False,
            random_mask=True,
        )

        self.assertEqual(torch.triu(total_mask), torch.zeros([features, features]))


if __name__ == "__main__":
    unittest.main()