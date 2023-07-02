import torch
from torch import nn

from nflows.utils.typechecks import validate_positive_int


class DistributionException(Exception):
    """Base exception class for distributions."""


class NoMeanException(DistributionException):
    """Exception to be thrown when a mean function doesn't exist."""


class Distribution(nn.Module):
    """Base class for all distribution objects."""

    def forward(self, *args):
        raise RuntimeError("Forward method cannot be called for a Distribution object.")

    def log_prob(self, inputs, context=None):
        """Calculate log probability under the distribution.

        Args:
            inputs (Tensor): Input variables.
            context (Tensor or None): Conditioning variables. If a Tensor, it must have the same
                number of rows as the inputs. If None, the context is ignored.

        Returns:
            Tensor: Log probability of the inputs given the context, with shape [input_size].
        """
        if context is not None:
            validate_shapes(inputs, context)

        return self.log_prob(inputs, context)

    def _log_prob(self, inputs, context):
        raise NotImplementedError()

    def sample(self, num_samples, context=None, batch_size=None):
        """Generates samples from the distribution. Samples can be generated in batches.

        Args:
            num_samples (int): Number of samples to generate.
            context (Tensor or None): Conditioning variables. If None, the context is ignored.
                Should have shape [context_size, ...], where ... represents a (context) feature
                vector of arbitrary shape. This will generate num_samples for each context item
                provided. The overall shape of the samples will then be
                [context_size, num_samples, ...].
            batch_size (int or None): Number of samples per batch. If None, all samples are generated
                in one batch.

        Returns:
            Tensor: Samples with shape [num_samples, ...] if context is None, or
                [context_size, num_samples, ...] if context is given, where ... represents a feature
                vector of arbitrary shape.
        """
        validate_positive_int(num_samples)
        validate_positive_int(batch_size, can_be_none=True)

        if context is not None:
            context = torch.as_tensor(context)

        if batch_size is None:
            return self.sample(num_samples, context)

        else:
            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples = [self.sample(batch_size, context) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self.sample(num_leftover, context))
            return torch.cat(samples, dim=0)

    def _sample(self, num_samples, context):
        raise NotImplementedError()

    def sample_and_log_prob(self, num_samples, context=None):
        """Generates samples from the distribution together with their log probability.

        Args:
            num_samples (int): Number of samples to generate.
            context (Tensor or None): Conditioning variables. If None, the context is ignored.
                Should have shape [context_size, ...], where ... represents a (context) feature
                vector of arbitrary shape. This will generate num_samples for each context item
                provided. The overall shape of the samples will then be
                [context_size, num_samples, ...].

        Returns:
            tuple: A tuple containing:
                - Tensor: Samples with shape [num_samples, ...] if context is None, or
                          [context_size, num_samples, ...] if context is given, where ... represents
                          a feature vector of arbitrary shape.
                - Tensor: Log probabilities of the samples, with shape
                          [num_samples, features] if context is None, or
                          [context_size, num_samples, ...] if context is given.
        """
        samples = self.sample(num_samples, context=context)

        if context is not None:
            samples = samples.view(samples.size(0), -1)
            context = context.unsqueeze(1).expand(-1, num_samples, -1)
            assert samples.shape[0] == context.shape[0]

        log_prob = self.log_prob(samples, context=context)

        if context is not None:
            samples = samples.view(-1, num_samples, context.size(-1))
            log_prob = log_prob.view(-1, num_samples)

        return samples, log_prob

    def mean(self, context=None):
        """Calculate the mean value under the distribution.

        Args:
            context (Tensor or None): Conditioning variables. If None, the context is ignored.

        Returns:
            Tensor: Mean value with shape [context_size, ...] if context is given,
                or [feature_size, ...] if context is None, where ... represents a
                feature vector of arbitrary shape.
        """
        if context is not None:
            validate_shapes(context)

        return self.mean(context)

    def _mean(self, context):
        raise NoMeanException()