import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

NOISY_LAYER_STD = 0.4
MU_RANGE = 1 / math.sqrt
BIAS_RANGE = 0

class BaseNet:
    """
    A base class for neural networks.
    """
    def __init__(self):
        pass

    def reset_noise(self):
        """
        Reset the noise for NoisyLinear layers.
        """
        pass


def layer_init(layer, w_scale=1.0):
    """
    Initialize the weights and biases of a layer.

    Args:
    - layer: a PyTorch layer object
    - w_scale: a scaling factor for the weights
    """
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, BIAS_RANGE)
    return layer

class NoisyLinear(nn.Module):
    """
    A NoisyLinear layer for use in neural networks. Adapted from
    https://github.com/saj1919/RL-Adventure/blob/master/5.noisy%20dqn.ipynb
    """
    def __init__(self, in_features, out_features, std_init=NOISY_LAYER_STD):
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
        The forward pass through the NoisyLinear layer.

        Args:
        - x: the input tensor

        Returns:
        - a tensor resulting from the linear transformation with noise
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
        Reset the weight and bias parameters of the NoisyLinear layer.
        """
        mu_range = MU_RANGE(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(BIAS_RANGE, BIAS_RANGE)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        """
        Reset the noise for the NoisyLinear layer.
        """
        self.noise_in.normal_(std=NOISY_LAYER_STD)
        self.noise_out_weight.normal_(std=NOISY_LAYER_STD)
        self.noise_out_bias.normal_(std=NOISY_LAYER_STD)

        self.weight_epsilon.copy_(self.transform_noise(self.noise_out_weight).ger(
            self.transform_noise(self.noise_in)))
        self.bias_epsilon.copy_(self.transform_noise(self.noise_out_bias)))

    def transform_noise(self, x):
        """
        Transform the noise for the NoisyLinear layer.

        Args:
        - x: the noise tensor

        Returns:
        - the transformed noise tensor
        """
        return x.sign().mul(x.abs().sqrt())

class NeuralNet(BaseNet):
    """
    A neural network with NoisyLinear layers.
    """
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.fc1 = layer_init(NoisyLinear(4, 128))
        self.fc2 = layer_init(NoisyLinear(128, 128))
        self.fc3 = layer_init(NoisyLinear(128, 2))

    def reset_noise(self):
        """
        Reset the noise for all NoisyLinear layers in the neural network.
        """
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()

    def forward(self, x):
        """
        The forward pass through the neural network.

        Args:
        - x: the input tensor

        Returns:
        - a tensor resulting from the forward pass through the neural network
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x