import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BaseNet:
    def __init__(self):
        pass

    def reset_noise(self):
        pass


def initialize_layer(layer, w_scale=1.0):
    """Initialize the weights and biases of a layer with scaling."""
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class NoisyLinear(nn.Module):
    """A fully connected layer with parameterized noise in inputs and outputs."""
    def __init__(self, in_features, out_features, std_init=0.4):
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

    def forward(self, x):
        """Forward pass with parameterized noise."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        """Initialize weights and biases with specific ranges."""
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self, std=0.5):
        """Reset the noise parameters with the given standard deviation."""
        self.noise_in.normal_(std=std)
        self.noise_out_weight.normal_(std=std)
        self.noise_out_bias.normal_(std=std)

        self.weight_epsilon.copy_(self.transform_noise(self.noise_out_weight).ger(
            self.transform_noise(self.noise_in)))
        self.bias_epsilon.copy_(self.transform_noise(self.noise_out_bias))

    def transform_noise(self, x):
        """Transform the noise with a signed square root function."""
        return x.sign().mul(x.abs().sqrt())