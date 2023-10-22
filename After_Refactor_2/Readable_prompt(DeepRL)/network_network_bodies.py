# Refactored code:

# Import necessary modules
import torch.nn as nn
import torch.nn.functional as F
from .network_utils import layer_init, NoisyLinear

# Define the NatureConvBody class
class NatureConvBody(nn.Module):
    def __init__(self, in_channels=4, noisy_linear=False):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        if noisy_linear:
            self.fc4 = NoisyLinear(7 * 7 * 64, self.feature_dim)
        else:
            self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        if self.noisy_linear:
            self.fc4.reset_noise()

    def forward(self, x):
        # Apply the convolutional layers and flatten the output
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.flatten(start_dim=1)
        # Apply the fully connected layer and return the result
        y = F.relu(self.fc4(y))
        return y


# Define the DDPGConvBody class
class DDPGConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(DDPGConvBody, self).__init__()
        self.feature_dim = 39 * 39 * 32
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

    def forward(self, x):
        # Apply the convolutional layers and flatten the output
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.flatten(start_dim=1)
        return y


# Define the FCBody class
class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu, noisy_linear=False):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        if noisy_linear:
            # Use NoisyLinear for fully connected layers that have noisy linear activation
            self.layers = nn.ModuleList(
                [NoisyLinear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        else:
            # Use nn.Linear for fully connected layers that don't have noisy linear activation
            self.layers = nn.ModuleList(
                [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        self.gate = gate
        self.feature_dim = dims[-1]
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        # Reset the noise in all the NoisyLinear layers if there are any
        if self.noisy_linear:
            for layer in self.layers:
                layer.reset_noise()

    def forward(self, x):
        # Apply fully connected layers with the specified activation function
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


# Define the DummyBody class
class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x