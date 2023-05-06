import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BaseNet(nn.Module):
    def reset_noise(self):
        pass

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super().__init__()

        mu_range = 1 / math.sqrt(in_features)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-mu_range, mu_range))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features).fill_(std_init / math.sqrt(in_features)))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-mu_range, mu_range))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features).fill_(std_init / math.sqrt(out_features)))

        self.register_buffer('weight_eps', torch.zeros(out_features, in_features))
        self.register_buffer('bias_eps', torch.zeros(out_features))

        self.reset_noise()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)

    def _apply_noise(self):
        noise_in = torch.randn(self.in_features)
        noise_out = torch.randn(self.out_features)

        self.weight_eps.copy_((noise_out.abs().sqrt() * noise_in.sign()).unsqueeze(1))
        self.bias_eps.copy_(noise_out)

    def reset_weights(self):
        self.weight_mu.data.uniform_(-1, 1).div_(math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-1, 1).div_(math.sqrt(self.in_features))

        self.weight_sigma.fill_(0.5 / math.sqrt(self.in_features))
        self.bias_sigma.fill_(0.5 / math.sqrt(self.out_features))

    def reset_noise(self):
        self._apply_noise()

class Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()

        self.fc1 = NoisyLinear(input_dim, hidden_dim)
        self.fc2 = NoisyLinear(hidden_dim, hidden_dim)
        self.fc3 = NoisyLinear(hidden_dim, output_dim)

    def forward(self, input):
        hidden = F.relu(self.fc1(input))
        hidden = F.relu(self.fc2(hidden))
        output = self.fc3(hidden)
        return output