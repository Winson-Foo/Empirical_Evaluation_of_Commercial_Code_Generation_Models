#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch.nn.functional as F


class NatureConvBody(nn.Module):
    def __init__(self, in_channels=4, noisy_linear=False):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        layers = []
        layers.append(layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)))
        layers.append(layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)))
        layers.append(layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)))
        self.conv_layers = nn.Sequential(*layers)
        if noisy_linear:
            self.fc4 = NoisyLinear(7 * 7 * 64, self.feature_dim)
        else:
            self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        if self.noisy_linear:
            self.fc4.reset_noise()

    def forward(self, x):
        y = self.conv_layers(x)
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y


class DdpgConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(DdpgConvBody, self).__init__()
        self.feature_dim = 39 * 39 * 32
        layers = []
        layers.append(layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)))
        layers.append(layer_init(nn.Conv2d(32, 32, kernel_size=3)))
        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        y = F.elu(self.conv_layers(x))
        y = y.view(y.size(0), -1)
        return y


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu, noisy_linear=False):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        layers = []
        for i in range(len(dims)-1):
            if noisy_linear:
                layers.append(NoisyLinear(dims[i], dims[i+1]))
            else:
                layers.append(layer_init(nn.Linear(dims[i], dims[i+1])))
            layers.append(gate)
        self.fc_layers = nn.Sequential(*layers)
        self.feature_dim = dims[-1]
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        if self.noisy_linear:
            for layer in self.fc_layers:
                if isinstance(layer, NoisyLinear):
                    layer.reset_noise()

    def forward(self, x):
        y = self.fc_layers(x)
        return y


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x