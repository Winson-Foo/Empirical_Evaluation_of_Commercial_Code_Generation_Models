# Refactored code:

# Import necessary libraries
import torch.nn.functional as F
from .network_utils import *


# Create a Class to define NatureConvBody
class NatureConvBody(nn.Module):
    def __init__(self, in_channels=4, noisy_linear=False):
        super(NatureConvBody, self).__init__()

        # Define feature dimension
        self.feature_dim = 512

        # Create Convolution layers
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))

        # Check if noisy linear model is to be implemented
        if noisy_linear:
            self.fc4 = NoisyLinear(7 * 7 * 64, self.feature_dim)
        else:
            self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))
        self.noisy_linear = noisy_linear

    # Function to reset noise
    def reset_noise(self):
        if self.noisy_linear:
            self.fc4.reset_noise()

    # Forward propagation function
    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y


# Create a Class to define DDPGConvBody
class DDPGConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(DDPGConvBody, self).__init__()

        # Define feature dimension
        self.feature_dim = 39 * 39 * 32

        # Create Convolution layers
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

    # Forward propagation function
    def forward(self, x):
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.view(y.size(0), -1)
        return y


# Create a Class to define FCBody
class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu, noisy_linear=False):
        super(FCBody, self).__init__()

        # Define necessary values 
        dims = (state_dim,) + hidden_units

        # Check if noisy linear model is to be implemented
        if noisy_linear:
            self.layers = nn.ModuleList(
                [NoisyLinear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        else:
            self.layers = nn.ModuleList(
                [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        # Set gate activation function
        self.gate = gate

        # Define feature dimension
        self.feature_dim = dims[-1]

        # Check if noisy linear model is to be implemented
        self.noisy_linear = noisy_linear

    # Function to reset noise
    def reset_noise(self):
        if self.noisy_linear:
            for layer in self.layers:
                layer.reset_noise()

    # Forward propagation function
    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


# Create a Class to define DummyBody
class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()

        # Define feature dimension
        self.feature_dim = state_dim

    # Forward propagation function
    def forward(self, x):
        return x