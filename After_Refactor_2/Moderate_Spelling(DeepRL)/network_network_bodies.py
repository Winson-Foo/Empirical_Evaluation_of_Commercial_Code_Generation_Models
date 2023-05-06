import torch.nn as nn
import torch.nn.functional as F
from .network_utils import layer_init, NoisyLinear


class NatureConvBody(nn.Module):
    """Convolutional neural network body for Deep Q Networks"""

    def __init__(self, in_channels=4, use_noisy_linear=False):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        if use_noisy_linear:
            self.fc4 = NoisyLinear(7 * 7 * 64, self.feature_dim)
        else:
            self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))
        self.use_noisy_linear = use_noisy_linear

    def reset_noise(self):
        """Reset the noise for the NoisyLinear layer, if applicable"""
        if self.use_noisy_linear:
            self.fc4.reset_noise()

    def forward(self, x):
        """Forward pass through the convolutional neural network"""
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y


class DDPGConvBody(nn.Module):
    """Convolutional neural network body for Deep Deterministic Policy Gradient"""

    def __init__(self, in_channels=4):
        super(DDPGConvBody, self).__init__()
        self.feature_dim = 39 * 39 * 32
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

    def forward(self, x):
        """Forward pass through the convolutional neural network"""
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.view(y.size(0), -1)
        return y


class FCBody(nn.Module):
    """Fully connected neural network body"""

    def __init__(self, state_dim, hidden_units=(64, 64), activation=F.relu, use_noisy_linear=False):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        if use_noisy_linear:
            self.layers = nn.ModuleList(
                [NoisyLinear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        else:
            self.layers = nn.ModuleList(
                [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        self.activation = activation
        self.feature_dim = dims[-1]
        self.use_noisy_linear = use_noisy_linear

    def reset_noise(self):
        """Reset the noise for the NoisyLinear layers, if applicable"""
        if self.use_noisy_linear:
            for layer in self.layers:
                layer.reset_noise()

    def forward(self, x):
        """Forward pass through the fully connected neural network"""
        for layer in self.layers:
            x = self.activation(layer(x))
        return x


class DummyBody(nn.Module):
    """Dummy neural network body that returns the input as the output"""

    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        """Return the input as the output"""
        return x