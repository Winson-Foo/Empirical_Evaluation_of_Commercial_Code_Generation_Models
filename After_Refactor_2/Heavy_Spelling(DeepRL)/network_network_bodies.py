import torch.nn as nn
import torch.nn.functional as F
from .network_utils import *


class ConvBody(nn.Module):
    """
    Base class to define a convolutional body network.
    """
    def __init__(self, in_channels, out_dim, layer_sizes=None, gating_function=F.relu, noisy_linear=False):
        super().__init__()
        self.gating_function = gating_function
        self.noisy_linear = noisy_linear
        self.layers = nn.ModuleList()
        
        layer_sizes = layer_sizes or [32, 64, 64]
        self.conv1 = layer_init(nn.Conv2d(in_channels, layer_sizes[0], kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(layer_sizes[0], layer_sizes[1], kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(layer_sizes[1], layer_sizes[2], kernel_size=3, stride=1))
        if noisy_linear:
            self.fc = NoisyLinear(self.get_flatten_size(), out_features=out_dim)
        else:
            self.fc = layer_init(nn.Linear(self.get_flatten_size(), out_features=out_dim))

    def get_flatten_size(self):
        # Get the size of the last conv layer's output flattened into a 1D array.
        test_tensor = torch.zeros((1, *self.input_size))
        output = self.conv3(self.conv2(self.conv1(test_tensor)))
        output_size = output.view(1, -1).size()[1]
        return output_size

    def reset_noise(self):
        if self.noisy_linear:
            self.fc.reset_noise()

    def forward(self, x):
        y = self.gating_function(self.conv1(x))
        y = self.gating_function(self.conv2(y))
        y = self.gating_function(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = self.gating_function(self.fc(y))
        return y


class NatureConvBody(ConvBody):
    """
    NatureConvBody extends ConvBody and uses the default layer sizes and gating function.
    """
    def __init__(self, in_channels=4, out_dim=512, noisy_linear=False):
        super().__init__(in_channels, out_dim, noisy_linear=noisy_linear)


class DDPGConvBody(ConvBody):
    """
    DDPGConvBody extends ConvBody and uses different layer sizes for the convolutional layers.
    """
    def __init__(self, in_channels=4, out_dim=39*39*32):
        super().__init__(
            in_channels,
            out_dim,
            layer_sizes=[32,32],
            gating_function=F.elu,
            noisy_linear=False
        )


class FCBody(nn.Module):
    """
    FCBody defines a fully-connected body network with multiple hidden layers.
    """
    def __init__(self, in_features, out_features, hidden_sizes=(64, 64), gating_function=F.relu, noisy_linear=False):
        super().__init__()

        self.gating_function = gating_function
        self.noisy_linear = noisy_linear
        self.layers = nn.ModuleList()

        sizes = (in_features,) + hidden_sizes + (out_features,)
        if noisy_linear:
            self.layers = nn.ModuleList(
                [NoisyLinear(sizes[i], sizes[i+1]) for i in range(len(sizes) - 1)]
            )
        else:
            self.layers = nn.ModuleList(
                [layer_init(nn.Linear(sizes[i], sizes[i+1])) for i in range(len(sizes) - 1)]
            )

    def reset_noise(self):
        if self.noisy_linear:
            for layer in self.layers:
                layer.reset_noise()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.gating_function(layer(x))
        x = self.layers[-1](x)
        return x


class DummyBody(nn.Module):
    """
    DummyBody simply passes through its input as output.
    """
    def __init__(self, in_features):
        super().__init__()
        self.out_features = in_features

    def forward(self, x):
        return x