import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BaseNetwork:
    """
    Base network class for all deep reinforcement learning models.
    """
    def __init__(self):
        pass

    def reset_noise(self):
        pass

def layer_init(layer, w_scale=1.0):
    """
    Initialize the weights and biases of a linear layer.
    """
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class NoisyLinear(nn.Module):
    """
    Linear layer with noise for exploration.
    """
    NOISY_LAYER_STD = 0.4

    def __init__(self, in_features, out_features, std_init=NOISY_LAYER_STD):
        """
        Initialize the NoisyLinear layer.
        """
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
        """
        Perform a forward pass through the NoisyLinear layer.
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        """
        Reset the weights and biases of the NoisyLinear layer.
        """
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        """
        Generate new noise vectors for the NoisyLinear layer.
        """
        self.noise_in.normal_(std=NoisyLinear.NOISY_LAYER_STD)
        self.noise_out_weight.normal_(std=NoisyLinear.NOISY_LAYER_STD)
        self.noise_out_bias.normal_(std=NoisyLinear.NOISY_LAYER_STD)

        self.weight_epsilon.copy_(self.transform_noise(self.noise_out_weight).ger(
            self.transform_noise(self.noise_in)))
        self.bias_epsilon.copy_(self.transform_noise(self.noise_out_bias))

    def transform_noise(self, x):
        """
        Transform the noise tensor for the NoisyLinear layer.
        """
        return x.sign().mul(x.abs().sqrt())