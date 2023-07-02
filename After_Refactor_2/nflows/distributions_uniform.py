import torch
from torch import distributions


class BoxUniform(distributions.Independent):
    def __init__(
        self,
        low: torch.Tensor,
        high: torch.Tensor,
        reinterpreted_batch_ndims: int = 1
    ):
        super().__init__(
            distributions.Uniform(low=low, high=high),
            reinterpreted_batch_ndims
        )


class MG1Uniform(distributions.Uniform):
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return super().log_prob(self._to_noise(value))

    def sample(self, sample_shape: torch.Size() = torch.Size()) -> torch.Tensor:
        return self._to_parameters(super().sample(sample_shape))

    def _to_parameters(self, noise: torch.Tensor) -> torch.Tensor:
        A_inv = torch.tensor([[1.0, 1, 0], [0, 1, 0], [0, 0, 1]])
        return noise @ A_inv

    def _to_noise(self, parameters: torch.Tensor) -> torch.Tensor:
        A = torch.tensor([[1.0, -1, 0], [0, 1, 0], [0, 0, 1]])
        return parameters @ A


class LotkaVolterraOscillating:
    def __init__(self):
        mean = torch.log(torch.tensor([0.01, 0.5, 1, 0.01]))
        sigma = 0.5
        covariance = sigma ** 2 * torch.eye(4)
        self.gaussian = distributions.MultivariateNormal(
            loc=mean, covariance_matrix=covariance
        )
        self.uniform = BoxUniform(
            low=-5 * torch.ones(4), high=2 * torch.ones(4)
        )
        self.log_normalizer = -torch.log(
            torch.erf((2 - mean) / sigma) - torch.erf((-5 - mean) / sigma)
        ).sum()

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        unnormalized_log_prob = (
            self.gaussian.log_prob(value)
            + self.uniform.log_prob(value)
        )

        return self.log_normalizer + unnormalized_log_prob

    def sample(self, sample_shape: torch.Size() = torch.Size()) -> torch.Tensor:
        num_remaining_samples = sample_shape[0]
        samples = []
        while num_remaining_samples > 0:
            candidate_samples = self.gaussian.sample((num_remaining_samples,))

            uniform_log_prob = self.uniform.log_prob(candidate_samples)

            accepted_samples = candidate_samples[~torch.isinf(uniform_log_prob)]
            samples.append(accepted_samples.detach())

            num_accepted = (~torch.isinf(uniform_log_prob)).sum().item()
            num_remaining_samples -= num_accepted

        # Aggregate collected samples.
        samples = torch.cat(samples)

        # Make sure we have the right amount.
        samples = samples[: sample_shape[0], ...]
        assert samples.shape[0] == sample_shape[0]

        return samples