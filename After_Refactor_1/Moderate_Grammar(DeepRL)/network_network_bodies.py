from .network_utils import *


class ConvBody(nn.Module):
    def __init__(self, in_channels=4, noisy_linear=False):
        super(ConvBody, self).__init__()
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        if self.noisy_linear:
            for layer in self.layers:
                layer.reset_noise()


class NatureConvBody(ConvBody):
    def __init__(self, in_channels=4, noisy_linear=False):
        super(NatureConvBody, self).__init__(in_channels, noisy_linear)
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        if self.noisy_linear:
            self.fc4 = NoisyLinear(7 * 7 * 64, self.feature_dim)
        else:
            self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y


class DDPGConvBody(ConvBody):
    def __init__(self, in_channels=4):
        super(DDPGConvBody, self).__init__(in_channels)
        self.feature_dim = 39 * 39 * 32
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

    def forward(self, x):
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.view(y.size(0), -1)
        return y


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu, noisy_linear=False):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        layers = []
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            if noisy_linear:
                layers.append(NoisyLinear(dim_in, dim_out))
            else:
                layers.append(layer_init(nn.Linear(dim_in, dim_out)))
        self.layers = nn.ModuleList(layers)

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
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x