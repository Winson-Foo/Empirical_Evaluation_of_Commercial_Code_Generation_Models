import torch.nn as nn
import torch.nn.functional as F
import math

class BaseNet:
    def __init__(self):
        pass

    def reset_noise(self):
        pass


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


# Adapted from https://github.com/saj1919/RL-Adventure/blob/master/5.noisy%20dqn.ipynb
class NoisyLinear(nn.Module):
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
        if self.training:
            weight = self.weight_mu + (self.weight_sigma.mul(self.weight_epsilon))
            bias = self.bias_mu + (self.bias_sigma.mul(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        bound = math.sqrt(3) * self.std_init

        nn.init.uniform_(self.weight_mu, -mu_range, mu_range)
        nn.init.uniform_(self.bias_mu, -mu_range, mu_range)
        nn.init.kaiming_uniform_(self.weight_sigma, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.bias_sigma, a=math.sqrt(5))
        
        self.weight_sigma.data.mul_(bound)
        self.bias_sigma.data.mul_(bound)

    def reset_noise(self):
        self.noise_in.normal_(0, std=Config.NOISY_LAYER_STD)
        self.noise_out_weight.normal_(0, std=Config.NOISY_LAYER_STD)
        self.noise_out_bias.normal_(0, std=Config.NOISY_LAYER_STD)

        self.weight_epsilon.copy_(self.transform_noise(self.noise_out_weight).ger(self.transform_noise(self.noise_in)))
        self.bias_epsilon.copy_(self.transform_noise(self.noise_out_bias))

    def transform_noise(self, x):
        x = x.sign().mul(x.abs().sqrt())
        return x