import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..utils import Config


class BaseNet:
    def __init__(self):
        pass

    def reset_noise(self):
        pass


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

        self._reset_parameters()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def _reset_parameters(self):
        nn.init.orthogonal_(self.weight_mu.data)
        nn.init.constant_(self.bias_mu.data, 0)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        self.noise_in.normal_(std=Config.NOISY_LAYER_STD)
        self.noise_out_weight.normal_(std=Config.NOISY_LAYER_STD)
        self.noise_out_bias.normal_(std=Config.NOISY_LAYER_STD)

        self.weight_epsilon.copy_(self._transform_noise(self.noise_out_weight).ger(
            self._transform_noise(self.noise_in)))
        self.bias_epsilon.copy_(self._transform_noise(self.noise_out_bias))

    def _transform_noise(self, x):
        return x.sign().mul(x.abs().sqrt())