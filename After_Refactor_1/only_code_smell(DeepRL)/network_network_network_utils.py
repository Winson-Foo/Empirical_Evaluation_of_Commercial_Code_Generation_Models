import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NoisyLinear(nn.Module):
    """
    A linear layer with added noise to the weights and biases during training.

    Adapted from https://github.com/saj1919/RL-Adventure/blob/master/5.noisy%20dqn.ipynb
    """
    def __init__(self, in_features, out_features, std_init=0.4):
        super().__init__()

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
        """
        Compute the forward pass of the layer.
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
        Reset the weights and biases to their initial values.
        """
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        """
        Reset the noise for the weights and biases.
        """
        self.noise_in.normal_(std=Config.NOISY_LAYER_STD)
        self.noise_out_weight.normal_(std=Config.NOISY_LAYER_STD)
        self.noise_out_bias.normal_(std=Config.NOISY_LAYER_STD)

        self.weight_epsilon.copy_(self.transform_noise(self.noise_out_weight).ger(
            self.transform_noise(self.noise_in)))
        self.bias_epsilon.copy_(self.transform_noise(self.noise_out_bias))

    def transform_noise(self, x):
        """
        Transform the noise using a specific function.
        """
        return x.sign().mul(x.abs().sqrt())


class BaseNet(nn.Module):
    """
    A base class for neural networks.
    """
    def __init__(self):
        super().__init__()

    def reset_noise(self):
        """
        Reset the noise in the neural network.
        """
        pass


def init_layer(layer, w_scale=1.0):
    """
    Initialize the weights and biases of a neural network layer.

    Args:
    - layer: the layer object
    - w_scale: the scaling factor for the weights
    """
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer