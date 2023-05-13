from .network_utils import *


class ConvBody(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super(ConvBody, self).__init__()
        self.feature_dim = feature_dim
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

    def forward(self, x):
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.view(y.size(0), -1)
        return y


class NatureConvBody(ConvBody):
    def __init__(self, in_channels=4, noisy_linear=False):
        super(NatureConvBody, self).__init__(in_channels, feature_dim=512)
        if noisy_linear:
            self.fc4 = NoisyLinear(7 * 7 * 32, self.feature_dim)
        else:
            self.fc4 = layer_init(nn.Linear(7 * 7 * 32, self.feature_dim))
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        if self.noisy_linear:
            self.fc4.reset_noise()


class DDPGConvBody(ConvBody):
    def __init__(self, in_channels=4):
        super(DDPGConvBody, self).__init__(in_channels, feature_dim=39 * 39 * 32)


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), activation_fn=F.relu, noisy_linear=False):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        self.activation_fn = activation_fn
        self.feature_dim = dims[-1]
        if noisy_linear:
            self.layers = nn.ModuleList(
                [NoisyLinear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        else:
            self.layers = nn.ModuleList(
                [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

    def reset_noise(self):
        if noisy_linear:
            for layer in self.layers:
                layer.reset_noise()

    def forward(self, x):
        for layer in self.layers:
            x = self.activation_fn(layer(x))
        return x


class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x