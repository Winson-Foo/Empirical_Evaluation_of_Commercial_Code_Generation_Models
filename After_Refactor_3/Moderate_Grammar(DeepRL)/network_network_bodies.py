from typing import Tuple, Optional
import torch.nn as nn
import torch.nn.functional as F
from .network_utils import init_weights


class ConvBody(nn.Module):
    """Base class for convolutional network bodies."""

    def __init__(self, in_channels: int, feature_dim: int, noisy_linear: bool = False):
        super(ConvBody, self).__init__()
        self.feature_dim = feature_dim
        self.conv1 = init_weights(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = init_weights(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = init_weights(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        if noisy_linear:
            self.fc4 = NoisyLinear(7 * 7 * 64, feature_dim)
        else:
            self.fc4 = init_weights(nn.Linear(7 * 7 * 64, feature_dim))
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        """Reset the noise in the network if using NoisyLinear module."""
        if self.noisy_linear:
            self.fc4.reset_noise()

    def forward(self, x) -> torch.Tensor:
        """Forward pass of the network."""
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y


class NatureConvBody(ConvBody):
    """Network body used in the DQN algorithm."""

    def __init__(self, in_channels: int = 4, noisy_linear: bool = False):
        super(NatureConvBody, self).__init__(in_channels, 512, noisy_linear)


class DDPGConvBody(ConvBody):
    """Network body used in the DDPG algorithm."""

    def __init__(self, in_channels: int = 4):
        super(DDPGConvBody, self).__init__(in_channels, 39 * 39 * 32)


class FCBody(nn.Module):
    """Fully connected network body."""

    def __init__(self, state_dim: int, hidden_units: Tuple[int, int] = (64, 64),
                 activation: Optional[nn.Module] = None, noisy_linear: bool = False):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        if noisy_linear:
            self.layers = nn.ModuleList(
                [NoisyLinear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        else:
            self.layers = nn.ModuleList(
                [init_weights(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        if activation is None:
            self.activation = nn.ReLU()
        else:
            self.activation = activation
        self.feature_dim = dims[-1]
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        """Reset the noise in the network if using NoisyLinear module."""
        if self.noisy_linear:
            for layer in self.layers:
                layer.reset_noise()

    def forward(self, x) -> torch.Tensor:
        """Forward pass of the network."""
        for layer in self.layers:
            x = self.activation(layer(x))
        return x


class DummyBody(nn.Module):
    """Dummy network body that passes through the input."""

    def __init__(self, state_dim: int):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x) -> torch.Tensor:
        """Forward pass of the network."""
        return x