import pytest

import torch
from torch import Tensor

from nflows.distributions import discrete


@pytest.fixture
def random_input_tensor(batch_size, *input_shape):
    return torch.randn(batch_size, *input_shape)


@pytest.fixture
def random_context_tensor(batch_size, *context_shape):
    return torch.randn(batch_size, *context_shape)


@pytest.fixture
def conditional_bernoulli_dist(input_shape):
    return discrete.ConditionalIndependentBernoulli(input_shape)


def test_conditional_bernoulli_dist_log_prob(random_input_tensor, random_context_tensor, conditional_bernoulli_dist):
    log_prob = conditional_bernoulli_dist.log_prob(random_input_tensor, context=random_context_tensor)
    assert isinstance(log_prob, Tensor)
    assert log_prob.shape == torch.Size([random_input_tensor.shape[0]])
    assert not torch.isnan(log_prob).any()
    assert not torch.isinf(log_prob).any()
    assert torch.all(log_prob <= 0.0)


def test_conditional_bernoulli_dist_sample(random_context_tensor, conditional_bernoulli_dist):
    num_samples = 10
    samples = conditional_bernoulli_dist.sample(num_samples, context=random_context_tensor)
    assert isinstance(samples, Tensor)
    assert samples.shape == torch.Size([random_context_tensor.shape[0], num_samples] + list(conditional_bernoulli_dist.get_input_shape()))
    assert not torch.isnan(samples).any()
    assert not torch.isinf(samples).any()
    binary = (samples == 1.0) | (samples == 0.0)
    assert torch.all(binary)


def test_conditional_bernoulli_dist_sample_and_log_prob(random_context_tensor, conditional_bernoulli_dist):
    num_samples = 10
    samples, log_prob = conditional_bernoulli_dist.sample_and_log_prob(num_samples, context=random_context_tensor)

    assert isinstance(samples, Tensor)
    assert isinstance(log_prob, Tensor)

    assert samples.shape == torch.Size([random_context_tensor.shape[0], num_samples] + list(conditional_bernoulli_dist.get_input_shape()))
    assert log_prob.shape == torch.Size([random_context_tensor.shape[0], num_samples])

    assert not torch.isnan(log_prob).any()
    assert not torch.isinf(log_prob).any()
    assert torch.all(log_prob <= 0.0)

    assert not torch.isnan(samples).any()
    assert not torch.isinf(samples).any()
    binary = (samples == 1.0) | (samples == 0.0)
    assert torch.all(binary)


def test_conditional_bernoulli_dist_mean(random_context_tensor, conditional_bernoulli_dist):
    means = conditional_bernoulli_dist.mean(context=random_context_tensor)
    assert isinstance(means, Tensor)
    assert means.shape == torch.Size([random_context_tensor.shape[0]] + list(conditional_bernoulli_dist.get_input_shape()))
    assert not torch.isnan(means).any()
    assert not torch.isinf(means).any()
    assert torch.all(means >= 0.0) and torch.all(means <= 1.0)