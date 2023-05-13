# Refactored code:

from .network_utils import *


class NatureConvBody(nn.Module):
    def __init__(self, in_channels=4, noisy_linear=False):
        super(NatureConvBody, self).__init__()

        # Initialize feature_dim
        self.feature_dim = 512
        # Initialize Conv2d layers with layer_init function
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        # Initialize FC4 layer differently based on the noisy_linear boolean value
        if noisy_linear:
            self.fc4 = NoisyLinear(7 * 7 * 64, self.feature_dim)
        else:
            self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))
        # Initialize the noisy_linear attribute
        self.noisy_linear = noisy_linear

    # Define a function to reset noise for the NoisyLinear layers
    def reset_noise(self):
        if self.noisy_linear:
            self.fc4.reset_noise()

    # Define the feedforward function
    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y


class DDPGConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(DDPGConvBody, self).__init__()

        # Initialize feature_dim
        self.feature_dim = 39 * 39 * 32
        # Initialize Conv2d layers with layer_init function
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

    # Define the feedforward function
    def forward(self, x):
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.view(y.size(0), -1)
        return y


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu, noisy_linear=False):
        super(FCBody, self).__init__()

        # Define dims, a tuple of the dimensions of inputs and outputs of each layer
        dims = (state_dim,) + hidden_units
        # Initialize the ModuleList layers with NoisyLinear or nn.Linear, based on the noisy_linear boolean value
        if noisy_linear:
            self.layers = nn.ModuleList(
                [NoisyLinear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        else:
            self.layers = nn.ModuleList(
                [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        self.gate = gate
        # Initialize feature_dim with the last dimension defined in dims
        self.feature_dim = dims[-1]
        # Initialize the noisy_linear attribute
        self.noisy_linear = noisy_linear

    # Define a function to reset noise for the NoisyLinear layers
    def reset_noise(self):
        if self.noisy_linear:
            for layer in self.layers:
                layer.reset_noise()

    # Define the feedforward function
    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()

        # Initialize feature_dim with the state dimension
        self.feature_dim = state_dim

    # Define the feedforward function
    def forward(self, x):
        return x