from torch.nn import functional as F
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.normalization import BatchNorm
from nflows.transforms.permutations import RandomPermutation, ReversePermutation


class MaskedAutoregressiveFlow(Flow):
    def __init__(
        self,
        input_features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        use_residual_blocks=True,
        use_random_masks=False,
        use_random_permutations=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm_within_layers=False,
        use_batch_norm_between_layers=False,
    ):

        if use_random_permutations:
            permutation_transform = RandomPermutation(input_features)
        else:
            permutation_transform = ReversePermutation(input_features)

        affine_transforms = self._create_affine_transforms(
            input_features,
            hidden_features,
            num_layers,
            num_blocks_per_layer,
            use_residual_blocks,
            use_random_masks,
            activation,
            dropout_probability,
            use_batch_norm_within_layers,
            use_batch_norm_between_layers,
        )

        transform = CompositeTransform(
            [permutation_transform] + affine_transforms
        )

        distribution = StandardNormal([input_features])

        super().__init__(transform=transform, distribution=distribution)

    def _create_affine_transforms(
        self,
        input_features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        use_residual_blocks,
        use_random_masks,
        activation,
        dropout_probability,
        use_batch_norm_within_layers,
        use_batch_norm_between_layers,
    ):
        transforms = []

        for _ in range(num_layers):
            if use_random_masks:
                transform = RandomPermutation(input_features)
            else:
                transform = ReversePermutation(input_features)

            affine_transform = MaskedAffineAutoregressiveTransform(
                features=input_features,
                hidden_features=hidden_features,
                num_blocks=num_blocks_per_layer,
                use_residual_blocks=use_residual_blocks,
                random_mask=use_random_masks,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm_within_layers,
            )

            transforms.append(transform)
            transforms.append(affine_transform)

            if use_batch_norm_between_layers:
                transforms.append(BatchNorm(input_features))

        return transforms