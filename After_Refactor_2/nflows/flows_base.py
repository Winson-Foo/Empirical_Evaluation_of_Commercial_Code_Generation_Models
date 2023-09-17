from inspect import signature
import torch

from nflows.distributions.base import Distribution
from nflows.utils import torchutils


class Flow(Distribution):
    """Base class for all flow objects."""

    def __init__(
        self,
        transform: Transform,
        distribution: Distribution,
        embedding_net: torch.nn.Module = None
    ):
        super().__init__()
        self.transform = transform
        self.distribution = distribution
        distribution_signature = signature(self.distribution.log_prob)
        distribution_arguments = list(distribution_signature.parameters.keys())
        self.context_used_in_base = 'context' in distribution_arguments
        if embedding_net is not None:
            assert isinstance(embedding_net, torch.nn.Module), (
                "embedding_net is not a nn.Module. "
                "If you want to use hard-coded summary features, "
                "please simply pass the encoded features and pass "
                "embedding_net=None"
            )
            self.embedding_net = embedding_net
        else:
            self.embedding_net = torch.nn.Identity()

    def log_prob(self, inputs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        embedded_context = self.embedding_net(context)
        noise, logabsdet = self.transform(inputs, context=embedded_context)
        if self.context_used_in_base:
            log_prob = self.distribution.log_prob(noise, context=embedded_context)
        else:
            log_prob = self.distribution.log_prob(noise)
        return log_prob + logabsdet

    def sample(self, num_samples: int, context: torch.Tensor) -> torch.Tensor:
        embedded_context = self.embedding_net(context)
        if self.context_used_in_base:
            noise = self.distribution.sample(num_samples, context=embedded_context)
        else:
            repeat_noise = self.distribution.sample(num_samples * embedded_context.shape[0])
            noise = torch.reshape(
                repeat_noise,
                (embedded_context.shape[0], -1, repeat_noise.shape[1])
            )

        if embedded_context is not None:
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, _ = self.transform.inverse(noise, context=embedded_context)

        if embedded_context is not None:
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])

        return samples

    def sample_and_log_prob(
        self, num_samples: int, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded_context = self.embedding_net(context)
        if self.context_used_in_base:
            noise, log_prob = self.distribution.sample_and_log_prob(
                num_samples, context=embedded_context
            )
        else:
            noise, log_prob = self.distribution.sample_and_log_prob(num_samples)

        if embedded_context is not None:
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, logabsdet = self.transform.inverse(noise, context=embedded_context)

        if embedded_context is not None:
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = torchutils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet

    def transform_to_noise(
        self, inputs: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        noise, _ = self.transform(inputs, context=self.embedding_net(context))
        return noise