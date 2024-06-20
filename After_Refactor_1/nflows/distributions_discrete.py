import torch
from torch import nn
from torch.nn import functional as F
import nflows.distributions.base as base
import nflows.utils as utils


class ConditionalIndependentBernoulli(base.Distribution):
    """An independent Bernoulli distribution whose parameters are functions of a context."""

    def __init__(self, shape, context_encoder=None):
        """Constructor.

        Args:
            shape (Union[list, tuple, torch.Size]): The shape of the input variables.
            context_encoder (callable, optional): Encodes the context to the distribution parameters.
                If None, defaults to the identity function.
        """
        super().__init__()
        self.shape = torch.Size(shape)
        self.context_encoder = context_encoder or nn.Identity()

    def compute_params(self, context):
        """Compute the logits from the context.

        Args:
            context (torch.Tensor): The context for encoding the distribution parameters.

        Returns:
            torch.Tensor: The logits of the distribution.
        """
        if context is None:
            raise ValueError("Context cannot be None.")

        logits = self.context_encoder(context)
        if logits.shape[0] != context.shape[0]:
            raise RuntimeError("The batch dimension of the parameters is inconsistent with the input.")

        return logits.reshape(logits.shape[0], *self.shape)

    def log_prob(self, inputs, context):
        """Compute the log probability of the inputs given the context.

        Args:
            inputs (torch.Tensor): The input values.
            context (torch.Tensor): The context for encoding the distribution parameters.

        Returns:
            torch.Tensor: The log probability of the inputs.
        """
        if inputs.shape[1:] != self.shape:
            raise ValueError(f"Expected input of shape {self.shape}, got {inputs.shape[1:]}")

        logits = self.compute_params(context)
        assert logits.shape == inputs.shape

        log_prob = -inputs * F.softplus(-logits) - (1.0 - inputs) * F.softplus(logits)
        log_prob = utils.torchutils.sum_except_batch(log_prob, num_batch_dims=1)
        return log_prob

    def sample(self, num_samples, context):
        """Generate samples from the distribution given the context.

        Args:
            num_samples (int): The number of samples to generate.
            context (torch.Tensor): The context for encoding the distribution parameters.

        Returns:
            torch.Tensor: The generated samples.
        """
        logits = self.compute_params(context)
        probs = torch.sigmoid(logits)
        probs = utils.torchutils.repeat_rows(probs, num_samples)

        context_size = context.shape[0]
        noise = torch.rand(context_size * num_samples, *self.shape)
        samples = (noise < probs).float()
        return utils.torchutils.split_leading_dim(samples, [context_size, num_samples])

    def mean(self, context):
        """Compute the mean of the distribution given the context.

        Args:
            context (torch.Tensor): The context for encoding the distribution parameters.

        Returns:
            torch.Tensor: The mean of the distribution.
        """
        logits = self.compute_params(context)
        return torch.sigmoid(logits)