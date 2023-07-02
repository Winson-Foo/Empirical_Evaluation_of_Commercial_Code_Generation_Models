import torch
from torch.nn import functional as F

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.nn import nets as nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import (
    AdditiveCouplingTransform,
    AffineCouplingTransform,
)
from nflows.transforms.normalization import BatchNorm


class SimpleRealNVP(Flow):
    def __init__(
        self,
        features: int,
        hidden_features: int = 64,
        num_layers: int = 5,
        num_blocks_per_layer: int = 2,
        use_volume_preserving: bool = False,
        activation: callable = F.relu,
        dropout_probability: float = 0.0,
        batch_norm_within_layers: bool = False,
        batch_norm_between_layers: bool = False,
    ):
        super().__init__(
            transform=CompositeTransform(self._create_layers(
                features,
                hidden_features,
                num_layers,
                num_blocks_per_layer,
                use_volume_preserving,
                activation,
                dropout_probability,
                batch_norm_within_layers,
                batch_norm_between_layers,
            )),
            distribution=StandardNormal([features]),
        )

    def _create_layers(
        self,
        features: int,
        hidden_features: int,
        num_layers: int,
        num_blocks_per_layer: int,
        use_volume_preserving: bool,
        activation: callable,
        dropout_probability: float,
        batch_norm_within_layers: bool,
        batch_norm_between_layers: bool,
    ):
        if use_volume_preserving:
            coupling_constructor = AdditiveCouplingTransform
        else:
            coupling_constructor = AffineCouplingTransform

        mask = self._create_mask(features)

        def create_transform_net(in_features, out_features):
            return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_features,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers,
            )

        layers = []
        for _ in range(num_layers):
            transform = coupling_constructor(
                mask=mask, transform_net_create_fn=create_transform_net
            )
            layers.append(transform)
            mask *= -1
            if batch_norm_between_layers:
                layers.append(BatchNorm(features=features))

        return layers

    def _create_mask(self, features: int):
        mask = torch.ones(features)
        mask[::2] = -1
        return mask