import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BaseNet:
    def __init__(self):
        pass

    def reset_noise(self):
        pass


def layer_init(layer: nn.Module, w_scale: float = 1.0) -> nn.Module:
    """Initialize a linear layer with orthogonal weight initialization and zero bias initialization."""
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.4):
        """A linear layer with noisy weights and biases for exploration."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the linear layer with noisy weights and biases."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        """Initialize the weights and biases with uniform distribution and standard deviation."""
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        """Reset the noise added to weights and biases for exploration."""
        self.noise_in.normal_(std=Config.NOISY_LAYER_STD)
        self.noise_out_weight.normal_(std=Config.NOISY_LAYER_STD)
        self.noise_out_bias.normal_(std=Config.NOISY_LAYER_STD)

        self.weight_epsilon.copy_(self.transform_noise(self.noise_out_weight).ger(
            self.transform_noise(self.noise_in)))
        self.bias_epsilon.copy_(self.transform_noise(self.noise_out_bias)))

    def transform_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Transform the noise added to weights and biases for exploration."""
        return x.sign().mul(x.abs().sqrt())