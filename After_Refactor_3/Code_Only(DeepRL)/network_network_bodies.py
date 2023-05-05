# Refactored code:

from .network_utils import *


class NatureConvBody(nn.Module):
    def __init__(self, in_channels=4, noisy_linear=False):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        if noisy_linear:
            self.fc4 = NoisyLinear(7 * 7 * 64, self.feature_dim)
        else:
            self.fc4 = nn.Linear(7 * 7 * 64, self.feature_dim)
        self.noisy_linear = noisy_linear
        self._initialize_layers()

    def _initialize_layers(self):
        layers = [self.conv1, self.conv2, self.conv3]
        for layer in layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)
        if not self.noisy_linear:
            nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='relu')
            nn.init.constant_(self.fc4.bias, 0)

    def reset_noise(self):
        if self.noisy_linear:
            self.fc4.reset_noise()

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
        self.feature_dim = 39 * 39 * 32
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self._initialize_layers()

    def _initialize_layers(self):
        layers = [self.conv1, self.conv2]
        for layer in layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.view(y.size(0), -1)
        return y


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu, noisy_linear=False):
        super(FCBody, self).__init__()
        self.dims = [state_dim] + list(hidden_units)
        self.layers = nn.ModuleList()
        for i in range(1, len(self.dims)):
            if noisy_linear:
                self.layers.append(NoisyLinear(self.dims[i-1], self.dims[i]))
            else:
                self.layers.append(nn.Linear(self.dims[i-1], self.dims[i]))
        self.gate = gate
        self.feature_dim = self.dims[-1]
        self.noisy_linear = noisy_linear
        self._initialize_layers()

    def _initialize_layers(self):
        for layer in self.layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)

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