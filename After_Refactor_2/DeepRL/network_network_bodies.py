from .network_utils import *


class ConvBody(nn.Module):
    def __init__(self, in_channels, feature_dim, layers):
        super(ConvBody, self).__init__()
        self.feature_dim = feature_dim
        self.layers = nn.ModuleList(layers)
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

    def reset_noise(self):
        for layer in self.layers:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

    def forward(self, x):
        y = self.conv1(x)
        y = F.elu(y)
        y = self.conv2(y)
        y = F.elu(y)
        y = y.view(y.size(0), -1)
        for layer in self.layers:
            y = F.relu(layer(y))
        return y


class NatureConvBody(ConvBody):
    def __init__(self, in_channels=4, noisy_linear=False):
        feature_dim = 512
        layers = []
        layers.append(layer_init(nn.Linear(7 * 7 * 64, feature_dim))) if not noisy_linear else \
            layers.append(NoisyLinear(7 * 7 * 64, feature_dim))
        super().__init__(in_channels, feature_dim, layers)


class DDPGConvBody(ConvBody):
    def __init__(self, in_channels=4):
        feature_dim = 39 * 39 * 32
        layers = []
        super().__init__(in_channels, feature_dim, layers)


class FCBody(nn.Module):
    HIDDEN_UNITS = (64, 64)
    
    def __init__(self, state_dim, hidden_units=HIDDEN_UNITS, gate=F.relu, noisy_linear=False):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        self.noisy_linear = noisy_linear
        self.feature_dim = dims[-1]
        self.layers = nn.ModuleList()
        for i in range(1, len(dims)):
            if not noisy_linear:
                self.layers.append(layer_init(nn.Linear(dims[i-1], dims[i])))
            else:
                self.layers.append(NoisyLinear(dims[i-1], dims[i]))

        self.gate = gate

    def reset_noise(self):
        for layer in self.layers:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x