import torch
from torch import nn
from inspect import signature
from nflows.distributions.base import Distribution
from nflows.utils import torchutils


class Flow(Distribution):
    """Base class for all flow objects."""

    def __init__(self, transform, distribution, embedding_net: nn.Module = None):
        """Constructor.

        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution: A `Distribution` object, the base distribution of the flow that
                generates the noise.
            embedding_net: A `nn.Module` which has trainable parameters to encode the
                context (condition). It is trained jointly with the flow.
        """
        super().__init__()
        self.transform = transform
        self.distribution = distribution
        distribution_signature = signature(self.distribution.log_prob)
        distribution_args = distribution_signature.parameters.keys()
        self.context_used_in_base = 'context' in distribution_args
        if embedding_net is not None:
            assert isinstance(embedding_net, nn.Module), (
                "embedding_net is not a nn.Module. "
                "If you want to use hard-coded summary features, "
                "please simply pass the encoded features and pass "
                "embedding_net=None"
            )
            self.embedding_net = embedding_net
        else:
            self.embedding_net = nn.Identity()

    def log_prob(self, inputs: torch.Tensor, context: torch.Tensor):
        embedded_context = self.embedding_net(context)
        noise, log_abs_det = self.transform(inputs, context=embedded_context)
        if self.context_used_in_base:
            log_prob = self.distribution.log_prob(noise, context=embedded_context)
        else:
            log_prob = self.distribution.log_prob(noise)
        return log_prob + log_abs_det

    def sample(self, num_samples: int, context: torch.Tensor):
        embedded_context = self.embedding_net(context)
        if self.context_used_in_base:
            noise = self.distribution.sample(num_samples, context=embedded_context)
        else:
            repeat_noise = self.distribution.sample(num_samples * embedded_context.shape[0])
            noise = torch.reshape(repeat_noise, (embedded_context.shape[0], -1, repeat_noise.shape[1]))

        return self._apply_transform_inverse(noise, embedded_context, num_samples)

    def sample_and_log_prob(self, num_samples: int, context: torch.Tensor):
        embedded_context = self.embedding_net(context)
        if self.context_used_in_base:
            noise, log_prob = self.distribution.sample_and_log_prob(num_samples, context=embedded_context)
        else:
            noise, log_prob = self.distribution.sample_and_log_prob(num_samples)

        return self._apply_transform_inverse(noise, embedded_context, num_samples), log_prob

    def transform_to_noise(self, inputs: torch.Tensor, context: torch.Tensor):
        embedded_context = self.embedding_net(context)
        noise, _ = self.transform(inputs, context=embedded_context)
        return noise

    def _apply_transform_inverse(self, noise: torch.Tensor, embedded_context: torch.Tensor, num_samples: int):
        if embedded_context is not None:
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(embedded_context, num_reps=num_samples)

        samples, log_abs_det = self.transform.inverse(noise, context=embedded_context)

        if embedded_context is not None:
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            log_abs_det = torchutils.split_leading_dim(log_abs_det, shape=[-1, num_samples])

        return samples