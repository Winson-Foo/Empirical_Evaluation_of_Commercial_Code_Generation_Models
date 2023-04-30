from .network_utils import *


class ConvBody(nn.Module):
    """
    Base class for Convolutional Neural Network bodies
    """
    def __init__(self, in_channels, layers, feature_dim, noisy_linear=False):
        super(ConvBody, self).__init__()
        self.feature_dim = feature_dim
        self.noisy_linear = noisy_linear
        self.conv_layers = nn.ModuleList(layers)

        if noisy_linear:
            self.fc = NoisyLinear(self.fc_in_dim(), self.feature_dim)
        else:
            self.fc = layer_init(nn.Linear(self.fc_in_dim(), self.feature_dim))

    def reset_noise(self):
        if self.noisy_linear:
            self.fc.reset_noise()

    def fc_in_dim(self):
        raise NotImplementedError

    def forward(self, x):
        for layer in self.conv_layers:
            x = F.relu(layer(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


class NatureConvBody(ConvBody):
    """
    Convolutional Neural Network body used by the Nature DQN agent
    """
    def __init__(self, in_channels=4, noisy_linear=False):
        layers = [layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
                  layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                  layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))]
        feature_dim = 512
        super(NatureConvBody, self).__init__(in_channels, layers, feature_dim, noisy_linear)

    def fc_in_dim(self):
        return 7 * 7 * 64


class DDPGConvBody(ConvBody):
    """
    Convolutional Neural Network body used by the DDPG agent
    """
    def __init__(self, in_channels=4):
        layers = [layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)),
                  layer_init(nn.Conv2d(32, 32, kernel_size=3))]
        feature_dim = 39 * 39 * 32
        super(DDPGConvBody, self).__init__(in_channels, layers, feature_dim)

    def fc_in_dim(self):
        return self.feature_dim


class FCBody(nn.Module):
    """
    Fully Connected Neural Network body
    """
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu, noisy_linear=False):
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
        if self.noisy_linear:
            for layer in self.layers:
                layer.reset_noise()

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class DummyBody(nn.Module):
    """
    Dummy neural network body used as a placeholder
    """
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x