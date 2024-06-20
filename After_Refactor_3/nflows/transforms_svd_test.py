def setup_transform(self):
    self.features = 3
    self.transform = SVDLinear(features=self.features, num_householder=4)
    self.transform.bias.data = torch.randn(self.features)  # Just so bias isn't zero.

    diagonal = torch.diag(torch.exp(self.transform.log_diagonal))
    orthogonal_1 = self.transform.orthogonal_1.matrix()
    orthogonal_2 = self.transform.orthogonal_2.matrix()
    self.weight = orthogonal_1 @ diagonal @ orthogonal_2
    self.weight_inverse = torch.inverse(self.weight)
    self.logabsdet = torchutils.logabsdet(self.weight)

    self.eps = 1e-5

def setUp(self):
    self.setup_transform()

def test_forward(self):
    batch_size = 10
    inputs = torch.randn(batch_size, self.features)
    outputs, logabsdet = self.transform.forward_no_cache(inputs)

    outputs_ref = inputs @ self.weight.t() + self.transform.bias
    logabsdet_ref = torch.full([batch_size], self.logabsdet.item())

    self.assert_tensor_is_good(outputs, [batch_size, self.features])
    self.assert_tensor_is_good(logabsdet, [batch_size])

    self.assertEqual(outputs, outputs_ref)
    self.assertEqual(logabsdet, logabsdet_ref)

def test_inverse(self):
    batch_size = 10
    inputs = torch.randn(batch_size, self.features)
    outputs, logabsdet = self.transform.inverse_no_cache(inputs)

    outputs_ref = (inputs - self.transform.bias) @ self.weight_inverse.t()
    logabsdet_ref = torch.full([batch_size], -self.logabsdet.item())

    self.assert_tensor_is_good(outputs, [batch_size, self.features])
    self.assert_tensor_is_good(logabsdet, [batch_size])

    self.assertEqual(outputs, outputs_ref)
    self.assertEqual(logabsdet, logabsdet_ref)