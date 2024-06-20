import torch.nn.functional as F
from nflows.distributions.base import Distribution
from nflows.nn.nde import MixtureOfGaussiansMADE


class MADEMoG(Distribution):
    def __init__(
        self,
        input_features,
        hidden_features,
        context_features,
        num_blocks=2,
        num_mixture_components=1,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        custom_initialization=False,
    ):
        super().__init__()
        self.mixture_of_gaussians_made = MixtureOfGaussiansMADE(
            input_features=input_features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            num_mixture_components=num_mixture_components,
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
            custom_initialization=custom_initialization,
        )

    def log_prob(self, inputs, context=None):
        return self.mixture_of_gaussians_made.log_prob(inputs, context=context)

    def sample(self, num_samples, context=None):
        return self.mixture_of_gaussians_made.sample(num_samples, context=context)


def main():
    pass


if __name__ == "__main__":
    main()