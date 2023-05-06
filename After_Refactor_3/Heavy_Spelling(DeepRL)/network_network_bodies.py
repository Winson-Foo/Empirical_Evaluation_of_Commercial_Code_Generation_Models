from .network_utils import *


class ConvBody(nn.Module):
    def __init__(self, in_channels, feature_dim, conv_layers):
        super(ConvBody, self).__init__()
        self.feature_dim = feature_dim
        self.conv_layers = nn.ModuleList(conv_layers)
        self.fc_layer = layer_init(nn.Linear(self._conv_output_size(in_channels), feature_dim))

    def reset_noise(self):
        for layer in self.conv_layers:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

    def forward(self, x):
        for layer in self.conv_layers:
            x = F.relu(layer(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_layer(x))

        return x

    def _conv_output_size(self, in_channels):
        test_input = torch.ones([1, in_channels, 84, 84])
        with torch.no_grad():
            conv_output = self.conv_layers(test_input)
            flattened_size = conv_output.view(1, -1).size(1)
        return flattened_size


class NatureConvBody(ConvBody):
    def __init__(self, in_channels=4, noisy_linear=False):
        conv_layers = [layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
                       layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                       layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))]

        feature_dim = 512
        if noisy_linear:
            fc_layer = NoisyLinear(7 * 7 * 64, feature_dim)
        else:
            fc_layer = layer_init(nn.Linear(7 * 7 * 64, feature_dim))

        conv_layers.append(fc_layer)
        super(NatureConvBody, self).__init__(in_channels, feature_dim, conv_layers)


class DDPGConvBody(ConvBody):
    def __init__(self, in_channels=4):
        conv_layers = [layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)),
                       layer_init(nn.Conv2d(32, 32, kernel_size=3))]

        feature_dim = 39 * 39 * 32
        super(DDPGConvBody, self).__init__(in_channels, feature_dim, conv_layers)


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu, noisy_linear=False):
        super(FCBody, self).__init__()

        dims = (state_dim,) + hidden_units
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            if noisy_linear:
                fc_layer = NoisyLinear(dims[i], dims[i+1])
            else:
                fc_layer = layer_init(nn.Linear(dims[i], dims[i+1]))
            self.layers.append(fc_layer)

        self.gate = gate
        self.feature_dim = dims[-1]
        self.noisy_linear = noisy_linear

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