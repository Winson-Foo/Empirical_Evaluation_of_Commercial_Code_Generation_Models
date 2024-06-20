import torch
from torch import nn
from nflows.utils import torchutils
import nflows.utils.typechecks as check


class NoMeanException(Exception):
    pass


class Distribution(nn.Module):
    def forward(self, *args):
        raise RuntimeError("Forward method cannot be called for a Distribution object.")

    def log_prob(self, inputs: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        inputs, context = torch.as_tensor(inputs), torch.as_tensor(context)
        assert context is None or inputs.shape[0] == context.shape[0]
        return self._log_prob(inputs, context)

    def _log_prob(self, inputs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def sample(self, num_samples: int, context: torch.Tensor = None, batch_size: int = None) -> torch.Tensor:
        assert check.is_positive_int(num_samples), "Number of samples must be a positive integer."
        context = torch.as_tensor(context) if context is not None else None

        if batch_size is None:
            return self._sample(num_samples, context)
        
        assert check.is_positive_int(batch_size), "Batch size must be a positive integer."
        num_batches = num_samples // batch_size
        num_leftover = num_samples % batch_size
        samples = [self._sample(batch_size, context) for _ in range(num_batches)]
        if num_leftover > 0:
            samples.append(self._sample(num_leftover, context))
        return torch.cat(samples, dim=0)

    def _sample(self, num_samples: int, context: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def sample_and_log_prob(
        self, num_samples: int, context: torch.Tensor = None
    ) -> (torch.Tensor, torch.Tensor):
        samples = self.sample(num_samples, context=context)

        if context is not None:
            samples = torchutils.merge_leading_dims(samples, num_dims=2)
            context = torchutils.repeat_rows(context, num_reps=num_samples)
            assert samples.shape[0] == context.shape[0]

        log_prob = self.log_prob(samples, context=context)

        if context is not None:
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            log_prob = torchutils.split_leading_dim(log_prob, shape=[-1, num_samples])

        return samples, log_prob