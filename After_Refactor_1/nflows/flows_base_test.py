class FlowTest(torchtestcase.TorchTestCase):
    def create_flow(self, input_shape):
        return base.Flow(
            transform=AffineScalarTransform(scale=2.0),
            distribution=StandardNormal(input_shape),
        )

    def assert_shape_and_type(self, actual, expected_shape, expected_type):
        self.assertEqual(actual.shape, expected_shape)
        self.assertEqual(actual.dtype, expected_type)

    def test_log_prob(self):
        batch_size = 10
        input_shape = [2, 3, 4]
        context_shape = [5, 6]
        flow = self.create_flow(input_shape)
        inputs = torch.randn(batch_size, *input_shape)
        maybe_context = torch.randn(batch_size, *context_shape)
        for context in [None, maybe_context]:
            with self.subTest(context=context):
                log_prob = flow.log_prob(inputs, context=context)
                self.assert_shape_and_type(log_prob, torch.Size([batch_size]), torch.float32)

    def test_sample(self):
        num_samples = 10
        context_size = 20
        input_shape = [2, 3, 4]
        context_shape = [5, 6]
        flow = self.create_flow(input_shape)
        maybe_context = torch.randn(context_size, *context_shape)
        for context in [None, maybe_context]:
            with self.subTest(context=context):
                samples = flow.sample(num_samples, context=context)
                if context is None:
                    expected_shape = torch.Size([num_samples] + input_shape)
                else:
                    expected_shape = torch.Size([context_size, num_samples] + input_shape)
                self.assert_shape_and_type(samples, expected_shape, torch.float32)

    def test_sample_and_log_prob(self):
        num_samples = 10
        input_shape = [2, 3, 4]
        flow = self.create_flow(input_shape)
        samples, log_prob_1 = flow.sample_and_log_prob(num_samples)
        log_prob_2 = flow.log_prob(samples)
        self.assert_shape_and_type(samples, torch.Size([num_samples] + input_shape), torch.float32)
        self.assert_shape_and_type(log_prob_1, torch.Size([num_samples]), torch.float32)
        self.assert_shape_and_type(log_prob_2, torch.Size([num_samples]), torch.float32)
        self.assertEqual(log_prob_1, log_prob_2)

    def test_sample_and_log_prob_with_context(self):
        num_samples = 10
        context_size = 20
        input_shape = [2, 3, 4]
        context_shape = [5, 6]
        flow = self.create_flow(input_shape)
        context = torch.randn(context_size, *context_shape)
        samples, log_prob = flow.sample_and_log_prob(num_samples, context=context)
        expected_shape = torch.Size([context_size, num_samples] + input_shape)
        self.assert_shape_and_type(samples, expected_shape, torch.float32)
        expected_shape = torch.Size([context_size, num_samples])
        self.assert_shape_and_type(log_prob, expected_shape, torch.float32)

    def test_transform_to_noise(self):
        batch_size = 10
        context_size = 20
        shape = [2, 3, 4]
        context_shape = [5, 6]
        flow = self.create_flow(shape)
        inputs = torch.randn(batch_size, *shape)
        maybe_context = torch.randn(context_size, *context_shape)
        for context in [None, maybe_context]:
            with self.subTest(context=context):
                noise = flow.transform_to_noise(inputs, context=context)
                self.assert_shape_and_type(noise, torch.Size([batch_size] + shape), torch.float32)


if __name__ == "__main__":
    unittest.main()