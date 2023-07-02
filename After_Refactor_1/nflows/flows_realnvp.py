import torch
from torch import nn
from torch.nn import functional as F

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.nn import nets as nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import AdditiveCouplingTransform, AffineCouplingTransform
from nflows.transforms.normalization import BatchNorm


class SimpleRealNVP(Flow):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_blocks_per_layer: int,
        use_volume_preserving: bool = False,
        activation: nn.Module = F.relu,
        dropout_probability: float = 0.0,
        batch_norm_within_layers: bool = False,
        batch_norm_between_layers: bool = False,
    ):
        super().__init__(
            transform=self._create_transform(input_size, hidden_size, num_layers,
                                             num_blocks_per_layer, use_volume_preserving,
                                             activation, dropout_probability,
                                             batch_norm_within_layers, batch_norm_between_layers),
            distribution=StandardNormal([input_size]),
        )

    def _create_transform(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_blocks_per_layer: int,
        use_volume_preserving: bool,
        activation: nn.Module,
        dropout_probability: float,
        batch_norm_within_layers: bool,
        batch_norm_between_layers: bool,
    ) -> CompositeTransform:
        if use_volume_preserving:
            coupling_constructor = AdditiveCouplingTransform
        else:
            coupling_constructor = AffineCouplingTransform

        mask = torch.ones(input_size)
        mask[::2] = -1

        def create_resnet(in_features, out_features):
            return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_size,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers,
            )

        layers = []
        for _ in range(num_layers):
            transform = coupling_constructor(
                mask=mask, transform_net_create_fn=create_resnet
            )
            layers.append(transform)
            mask *= -1
            if batch_norm_between_layers:
                layers.append(BatchNorm(features=input_size))

        return CompositeTransform(layers)