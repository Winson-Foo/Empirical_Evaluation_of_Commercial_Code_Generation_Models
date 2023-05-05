import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..utils import *


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.zeros((out_features, in_features)))
        nn.init.uniform_(self.weight_mu, -1 / math.sqrt(in_features), 1 / math.sqrt(in_features))

        self.weight_sigma = nn.Parameter(torch.ones((out_features, in_features))) * std_init / math.sqrt(in_features)

        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        nn.init.uniform_(self.bias_mu, -1 / math.sqrt(in_features), 1 / math.sqrt(in_features))

        self.bias_sigma = nn.Parameter(torch.ones(out_features)) * std_init / math.sqrt(in_features)

        self.register_buffer('noise_in', torch.zeros(in_features))
        self.register_buffer('noise_out_weight', torch.zeros(out_features, in_features))
        self.register_buffer('noise_out_bias', torch.zeros(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.noise_out_weight
            bias = self.bias_mu + self.bias_sigma * self.noise_out_bias
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        pass

    def reset_noise(self):
        self.noise_in.normal_(0, 1)
        self.noise_out_weight.normal_(0, 1)
        self.noise_out_bias.normal_(0, 1)
        self.weight_epsilon = self.transform_noise(self.noise_out_weight).mul(self.transform_noise(self.noise_in))
        self.bias_epsilon = self.transform_noise(self.noise_out_bias)

    def transform_noise(self, x):
        return x.sign().mul(x.abs().sqrt())


class BaseNet:
    def __init__(self):
        pass

    def reset_noise(self):
        pass