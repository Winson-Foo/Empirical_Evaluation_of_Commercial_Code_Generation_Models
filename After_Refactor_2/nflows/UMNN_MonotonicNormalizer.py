import torch
import torch.nn as nn

from UMNN import NeuralIntegral, ParallelNeuralIntegral


def flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


class ELUPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.elu = nn.ELU()

    def forward(self, x):
        return self.elu(x) + 1.


class IntegrandNet(nn.Module):
    def __init__(self, hidden_sizes, cond_in):
        super(IntegrandNet, self).__init__()
        layer_input_sizes = [1 + cond_in] + hidden_sizes
        layer_output_sizes = hidden_sizes + [1]

        layers = []
        for input_size, output_size in zip(layer_input_sizes, layer_output_sizes):
            layers += [nn.Linear(input_size, output_size), nn.ReLU()]

        layers.pop()
        layers.append(ELUPlus())
        self.net = nn.Sequential(*layers)

    def forward(self, x, h):
        batch_size, input_dim = x.shape
        x = torch.cat((x, h), 1)
        x_he = x.view(batch_size, -1, input_dim).transpose(1, 2).contiguous().view(batch_size * input_dim, -1)
        y = self.net(x_he).view(batch_size, -1)
        return y


class MonotonicNormalizer(nn.Module):
    def __init__(self, integrand_net, cond_size, nb_steps=20, solver="CC"):
        super(MonotonicNormalizer, self).__init__()
        if isinstance(integrand_net, list):
            self.integrand_net = IntegrandNet(integrand_net, cond_size)
        else:
            self.integrand_net = integrand_net
        self.solver = solver
        self.nb_steps = nb_steps

    def forward(self, x, h, context=None):
        x0 = torch.zeros(x.shape).to(x.device)
        xT = x
        z0 = h[:, :, 0]
        h = h.permute(0, 2, 1).contiguous().view(x.shape[0], -1)
        
        if self.solver == "CC":
            z = NeuralIntegral.apply(x0, xT, self.integrand_net, flatten(self.integrand_net.parameters()),
                                     h, self.nb_steps) + z0
        elif self.solver == "CCParallel":
            z = ParallelNeuralIntegral.apply(x0, xT, self.integrand_net,
                                             flatten(self.integrand_net.parameters()),
                                             h, self.nb_steps) + z0
        else:
            return None

        return z, self.integrand_net(x, h)

    def inverse_transform(self, z, h, context=None):
        x_max = torch.ones_like(z) * 20
        x_min = -torch.ones_like(z) * 20
        z_max, _ = self.forward(x_max, h, context)
        z_min, _ = self.forward(x_min, h, context)

        for _ in range(25):
            x_middle = (x_max + x_min) / 2
            z_middle, _ = self.forward(x_middle, h, context)
            left = (z_middle > z).float()
            right = 1 - left
            x_max = left * x_middle + right * x_max
            x_min = right * x_middle + left * x_min
            z_max = left * z_middle + right * z_max
            z_min = right * z_middle + left * z_min

        return (x_max + x_min) / 2