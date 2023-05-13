# Refactored codebase
# Copyright (C) 2017 Shangtong Zhang
# Permission given to modify the code as long as you keep this declaration at the top

from .network_utils import *

class NatureConvBody(nn.Module):
    """
    A nature convolutional network body architecture.
    """

    def __init__(self, in_channels=4, noisy_linear=False):
        """
        Initialize the NatureConvBody.

        Arg:
        - in_channels: number of input channels, default: 4
        - noisy_linear: optionally use NoisyLinear layers, default: False
        """
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
        """
        Reset the noise in NoisyLinear layers, if any.
        """
        if self.noisy_linear:
            self.fc4.reset_noise()

    def forward(self, x):
        """
        Pass the input tensor x through the convolutional network and return the output tensor.
        """
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y


class DDPGConvBody(nn.Module):
    """
    A deep deterministic policy gradient convolutional network body architecture.
    """

    def __init__(self, in_channels=4):
        """
        Initialize the DDPGConvBody.

        Arg:
        - in_channels: number of input channels, default: 4
        """
        super(DDPGConvBody, self).__init__()
        self.feature_dim = 39 * 39 * 32
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

    def forward(self, x):
        """
        Pass the input tensor x through the convolutional network and return the output tensor.
        """
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.view(y.size(0), -1)
        return y


class FCBody(nn.Module):
    """
    A fully connected neural network body architecture.
    """

    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu, noisy_linear=False):
        """
        Initialize the FCBody.

        Arg:
        - state_dim: dimension of the state space
        - hidden_units: a tuple containing the number of neurons in each hidden layer, default: (64, 64)
        - gate: activation function, default: F.relu
        - noisy_linear: optionally use NoisyLinear layers, default: False
        """
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        if noisy_linear:
            self.layers = nn.ModuleList(
                [NoisyLinear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        else:
            self.layers = nn.ModuleList(
                [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        self.gate = gate
        self.feature_dim = dims[-1]
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        """
        Reset the noise in NoisyLinear layers, if any.
        """
        if self.noisy_linear:
            for layer in self.layers:
                layer.reset_noise()

    def forward(self, x):
        """
        Pass the input tensor x through the neural network and return the output tensor.
        """
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class DummyBody(nn.Module):
    """
    A dummy neural network body architecture that returns the input tensor as is.
    """

    def __init__(self, state_dim):
        """
        Initialize the DummyBody.

        Arg:
        - state_dim: dimension of the state space
        """
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        """
        Return the input tensor x as is.
        """
        return x