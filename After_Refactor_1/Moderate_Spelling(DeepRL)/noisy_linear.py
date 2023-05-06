# In noisy_linear.py module
import torch.nn as nn
import torch.nn.functional as F
import math
from ..utils import *

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super(NoisyLinear, self).__init__(in_features, out_features)

        self.std_init = 0.4
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
        super(NoisyLinear, self).reset_parameters()
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
        self.noise_in = torch.zeros(self.in_features)
        self.noise_out_weight = torch.zeros(self.out_features)
        self.noise_out_bias = torch.zeros(self.out_features)

    def reset_noise(self):
        self.noise_in.normal_(std=Config.NOISY_LAYER_STD)
        self.noise_out_weight.normal_(std=Config.NOISY_LAYER_STD)
        self.noise_out_bias.normal_(std=Config.NOISY_LAYER_STD)

        self.weight_epsilon.copy_(self.transform_noise(self.noise_out_weight).ger(
            self.transform_noise(self.noise_in)))
        self.bias_epsilon.copy_(self.transform_noise(self.noise_out_bias)))

    def transform_noise(self, x):
        return x.sign().mul(x.abs().sqrt())
