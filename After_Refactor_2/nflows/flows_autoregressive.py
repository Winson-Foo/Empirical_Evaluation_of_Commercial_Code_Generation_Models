## File: nflows/distributions/normal.py ##
"""Module for the Normal distribution."""

# Import statements (alphabetically)
...

# Class definition
class StandardNormal:
    def __init__(self, features: int):
        self.features = features

    # Rest of the class implementation...

## File: nflows/transforms/base.py ##
"""Module for base transform classes."""

# Import statements (alphabetically)
...

# Class definition
class CompositeTransform:
    def __init__(self, layers: List):
        self.layers = layers

    # Rest of the class implementation...

## File: nflows/transforms/normalization.py ##
"""Module for normalization transforms."""

# Import statements (alphabetically)
...

# Class definition
class BatchNorm:
    def __init__(self, features: int):
        self.features = features

    # Rest of the class implementation...

## File: nflows/transforms/permutations.py ##
"""Module for permutation transforms."""

# Import statements (alphabetically)
...

# Class definition
class RandomPermutation:
    def __init__(self, features: int):
        self.features = features

    # Rest of the class implementation...

class ReversePermutation:
    def __init__(self, features: int):
        self.features = features

    # Rest of the class implementation...

## File: nflows/transforms/autoregressive.py ##
"""Module for autoregressive transforms."""

# Import statements (alphabetically)
...

# Class definition
class MaskedAffineAutoregressiveTransform:
    def __init__(
        self,
        features: int,
        hidden_features: int,
        num_blocks: int,
        use_residual_blocks: bool,
        random_mask: bool,
        activation: Callable,
        dropout_probability: float,
        use_batch_norm: bool,
    ):
        self.features = features
        self.hidden_features = hidden_features
        self.num_blocks = num_blocks
        self.use_residual_blocks = use_residual_blocks
        self.random_mask = random_mask
        self.activation = activation
        self.dropout_probability = dropout_probability
        self.use_batch_norm = use_batch_norm

    # Rest of the class implementation...

## File: nflows/flows/autoregressive.py ##
"""Module for autoregressive flow classes."""

# Import statements (alphabetically)
...

# Class definition
class MaskedAutoregressiveFlow:
    def __init__(
        self,
        features: int,
        hidden_features: int,
        num_layers: int,
        num_blocks_per_layer: int,
        use_residual_blocks: bool = True,
        use_random_masks: bool = False,
        use_random_permutations: bool = False,
        activation: Callable = F.relu,
        dropout_probability: float = 0.0,
        batch_norm_within_layers: bool = False,
        batch_norm_between_layers: bool = False,
    ):
        if use_random_permutations:
            permutation_constructor = RandomPermutation
        else:
            permutation_constructor = ReversePermutation

        layers = []
        for _ in range(num_layers):
            layers.append(permutation_constructor(features))
            layers.append(
                MaskedAffineAutoregressiveTransform(
                    features=features,
                    hidden_features=hidden_features,
                    num_blocks=num_blocks_per_layer,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=use_random_masks,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=batch_norm_within_layers,
                )
            )
            if batch_norm_between_layers:
                layers.append(BatchNorm(features))

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal(features),
        )