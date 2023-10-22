import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from ..utils import Config


class BaseNet:
    def __init__(self):
        pass

    def reset_noise(self):
        pass


def layer_init(layer):
    nn.init.orthogonal_(layer.weight.data)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.zeros((out_features, in_features)))
        self.weight_sigma = nn.Parameter(torch.zeros((out_features, in_features)))
        self.register_buffer('weight_epsilon', torch.zeros((out_features, in_features)))

        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_sigma = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('bias_epsilon', torch.zeros(out_features))

        self.register_buffer('noise_in', torch.zeros(in_features))
        self.register_buffer('noise_out_weight', torch.zeros(out_features))
        self.register_buffer('noise_out_bias', torch.zeros(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        nn.init.uniform_(self.weight_mu.data, -mu_range, mu_range)
        nn.init.uniform_(self.bias_mu.data, -mu_range, mu_range)
        nn.init.constant_(self.weight_sigma.data, self.std_init / math.sqrt(self.weight_sigma.size(1)))
        nn.init.constant_(self.bias_sigma.data, self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        nn.init.normal_(self.noise_in, std=Config.NOISY_LAYER_STD)
        nn.init.normal_(self.noise_out_weight, std=Config.NOISY_LAYER_STD)
        nn.init.normal_(self.noise_out_bias, std=Config.NOISY_LAYER_STD)

        self.weight_epsilon.copy_(self.transform_noise(self.noise_out_weight).ger(
            self.transform_noise(self.noise_in)))
        self.bias_epsilon.copy_(self.transform_noise(self.noise_out_bias))

    def transform_noise(self, x):
        return x.sign().mul(x.abs().sqrt())