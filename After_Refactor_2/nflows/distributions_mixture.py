from typing import Optional
import torch
from torch.nn import functional as F

from nflows.distributions.base import Distribution
from nflows.nn.nde import MixtureOfGaussiansMADE


class MADEMoG(Distribution):
    def __init__(
        self,
        features: int,
        hidden_features: int,
        context_features: int,
        num_blocks: int = 2,
        num_mixture_components: int = 1,
        use_residual_blocks: bool = True,
        random_mask: bool = False,
        activation: Optional[torch.nn.Module] = F.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
        custom_initialization: bool = False
    ):
        super().__init__()

        self.made = MixtureOfGaussiansMADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            num_mixture_components=num_mixture_components,
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
            custom_initialization=custom_initialization
        )

    def log_prob(self, inputs: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.made.log_prob(inputs, context=context)

    def sample(self, num_samples: int, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.made.sample(num_samples, context=context)


if __name__ == "__main__":
    pass