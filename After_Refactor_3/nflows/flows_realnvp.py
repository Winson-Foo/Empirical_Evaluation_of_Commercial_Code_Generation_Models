import torch
from torch.nn import functional as F
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.nn import nets as nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import AdditiveCouplingTransform, AffineCouplingTransform
from nflows.transforms.normalization import BatchNorm


class SimpleRealNVP(Flow):
    def __init__(self, features, hidden_features, num_layers, num_blocks_per_layer,
                 use_volume_preserving=False, activation=F.relu, dropout_probability=0.0,
                 batch_norm_within_layers=False, batch_norm_between_layers=False):
        coupling_constructor = (AdditiveCouplingTransform if use_volume_preserving
                                else AffineCouplingTransform)

        mask = torch.ones(features)
        mask[::2] = -1

        layers = self.create_layers(features, hidden_features, num_layers, num_blocks_per_layer,
                                    activation, dropout_probability, batch_norm_within_layers,
                                    batch_norm_between_layers, mask, coupling_constructor)

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([features]),
        )

    def create_layers(self, features, hidden_features, num_layers, num_blocks_per_layer,
                      activation, dropout_probability, batch_norm_within_layers,
                      batch_norm_between_layers, mask, coupling_constructor):
        def create_resnet(in_features, out_features):
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
            transform = coupling_constructor(mask=mask, transform_net_create_fn=create_resnet)
            layers.append(transform)
            mask *= -1
            if batch_norm_between_layers:
                layers.append(BatchNorm(features=features))

        return layers