import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, std_init=0.4):
        super().__init__(in_features, out_features)

        self.std_init = std_init
        self.weight_sigma = nn.Parameter(torch.ones((out_features, in_features)), requires_grad=True)
        self.bias_sigma = nn.Parameter(torch.ones(out_features), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def forward(self, x):
        weight_epsilon = torch.Tensor(self.weight_sigma.size()).normal_()
        bias_epsilon = torch.Tensor(self.bias_sigma.size()).normal_()
        if self.training:
            weight = self.weight + self.weight_sigma * weight_epsilon
            bias = self.bias + self.bias_sigma * bias_epsilon
        else:
            weight = self.weight
            bias = self.bias
        return F.linear(x, weight, bias)


# Constants
NOISY_LAYER_STD = 0.4


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = layer_init(NoisyLinear(2, 128))
        self.fc2 = layer_init(NoisyLinear(128, 128))
        self.fc3 = layer_init(NoisyLinear(128, 1))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    x = torch.rand((1, 2))
    net = MyNet()
    y = net(x)
    print(y)