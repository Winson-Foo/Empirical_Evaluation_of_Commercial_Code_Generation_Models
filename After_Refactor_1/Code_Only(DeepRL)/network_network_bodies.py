# Refactored code:

from .network_utils import *

class NatureConvBody(nn.Module):
    """
    Convolutional network module.
    """
    def __init__(self, in_channels=4, noisy_linear=False):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        modules = []
        sizes = [in_channels, 32, 64, 64]
        for i in range(len(sizes) - 1):
            modules.append(nn.Conv2d(sizes[i], sizes[i+1], kernel_size=[8, 4, 3][i], stride=[4, 2, 1][i]))
            modules.append(nn.ReLU())
        if noisy_linear:
            modules.append(NoisyLinear(7 * 7 * 64, self.feature_dim))
        else:
            modules.append(nn.Linear(7 * 7 * 64, self.feature_dim))
        self.noisy_linear = noisy_linear
        self.feature_extractor = nn.Sequential(*modules)

    def reset_noise(self):
        if self.noisy_linear:
            self.feature_extractor[-1].reset_noise()

    def forward(self, x):
        x = self.feature_extractor(x)
        return x

class DDPGConvBody(nn.Module):
    """
    Convolutional network module.
    """
    def __init__(self, in_channels=4):
        super(DDPGConvBody, self).__init__()
        self.feature_dim = 39 * 39 * 32
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ELU()
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return x

class FCBody(nn.Module):
    """
    Fully connected network module.
    """
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu, noisy_linear=False):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        modules = []
        for i in range(len(dims) - 1):
            if noisy_linear:
                modules.append(NoisyLinear(dims[i], dims[i+1]))
            else:
                modules.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                modules.append(gate)
        self.noisy_linear = noisy_linear
        self.feature_extractor = nn.Sequential(*modules)
        self.feature_dim = dims[-1]

    def reset_noise(self):
        if self.noisy_linear:
            for module in self.feature_extractor:
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

    def forward(self, x):
        x = self.feature_extractor(x)
        return x

class DummyBody(nn.Module):
    """
    Dummy network module that returns unmodified inputs.
    """
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x