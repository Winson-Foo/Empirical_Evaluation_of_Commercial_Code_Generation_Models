import torch
from torch.nn import functional as F

from nflows.distributions.base import Distribution
from nflows.utils import torchutils


class ConditionalIndependentBernoulli(Distribution):
    """An independent Bernoulli whose parameters are functions of a context."""

    def __init__(self, shape, context_encoder=None):
        """Constructor.

        Args:
            shape: list, tuple or torch.Size, the shape of the input variables.
            context_encoder: callable or None, encodes the context to the distribution parameters.
                If None, defaults to the identity function.
        """
        super().__init__()
        self._shape = torch.Size(shape)
        if context_encoder is None:
            self._context_encoder = lambda x: x
        else:
            self._context_encoder = context_encoder

    def _compute_logits(self, context):
        """Compute the logits from context."""
        if context is None:
            raise ValueError("Context can't be None.")

        logits = self._context_encoder(context)
        if logits.shape[0] != context.shape[0]:
            raise RuntimeError("The batch dimension of the parameters is inconsistent with the input.")
        return logits.reshape(logits.shape[0], *self._shape)

    def _compute_log_prob(self, inputs, logits):
        """Compute the log probability of the inputs given the logits."""
        if inputs.shape[1:] != self._shape:
            raise ValueError("Expected input of shape {}, got {}".format(self._shape, inputs.shape[1:]))

        log_prob = -inputs * F.softplus(-logits) - (1.0 - inputs) * F.softplus(logits)
        log_prob = torchutils.sum_except_batch(log_prob, num_batch_dims=1)
        return log_prob

    def _generate_samples(self, num_samples, probs):
        """Generate samples from the distribution using the given probabilities."""
        context_size = probs.shape[0]
        noise = torch.rand(context_size * num_samples, *self._shape)
        samples = (noise < probs).float()
        return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def _log_prob(self, inputs, context):
        logits = self._compute_logits(context)
        assert logits.shape == inputs.shape
        return self._compute_log_prob(inputs, logits)

    def _sample(self, num_samples, context):
        logits = self._compute_logits(context)
        probs = torch.sigmoid(logits)
        probs = torchutils.repeat_rows(probs, num_samples)
        return self._generate_samples(num_samples, probs)

    def _mean(self, context):
        logits = self._compute_logits(context)
        return torch.sigmoid(logits)