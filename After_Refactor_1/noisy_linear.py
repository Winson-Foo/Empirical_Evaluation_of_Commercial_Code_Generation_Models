
import torch.nn as nn
import torch.nn.functional as F
import math
from ..utils import Config

class NoisyLinear(nn.Module):
    """
    A NoisyLinear module for use in Q-learning models.
    """
    def __init__(self, in_features, out_features, std_init=Config.NOISY_LAYER_STD):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.zeros((out_features, in_features)), requires_grad=True)
        self.weight_sigma = nn.Parameter(torch.zeros((out_features, in_features)), requires_grad=True)
        self.register_buffer('weight_epsilon', torch.zeros((out_features, in_features)))

        self.bias_mu = nn.Parameter(torch.zeros(out_features), requires_grad=True)
        self.bias_sigma = nn.Parameter(torch.zeros(out_features), requires_grad=True)
        self.register_buffer('bias_epsilon', torch.zeros(out_features))

        self.register_buffer('noise_in', torch.zeros(in_features))
        self.register_buffer('noise_out_weight', torch.zeros(out_features))
        self.register_buffer('noise_out_bias', torch.zeros(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, input):
        """
        Forward pass through the NoisyLinear module.
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(input, weight, bias)

    def reset_parameters(self):
        """
        Reset the parameters of the NoisyLinear module.
        """
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        """
        Reset the noise parameters of the NoisyLinear module.
        """
        self.noise_in.normal_(std=Config.NOISY_LAYER_STD)
        self.noise_out_weight.normal_(std=Config.NOISY_LAYER_STD)
        self.noise_out_bias.normal_(std=Config.NOISY_LAYER_STD)

        self.weight_epsilon.copy_(self.transform_noise(self.noise_out_weight).ger(
            self.transform_noise(self.noise_in)))
        self.bias_epsilon.copy_(self.transform_noise(self.noise_out_bias))

    def transform_noise(self, x):
        """
        Transform the input noise `x` as part of the NoisyLinear module.
        """
        return x.sign().mul(x.abs().sqrt())