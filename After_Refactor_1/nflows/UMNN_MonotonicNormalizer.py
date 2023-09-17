import torch
import torch.nn as nn

class ELUPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.elu = nn.ELU()

    def forward(self, x):
        return self.elu(x) + 1.


class IntegrandNet(nn.Module):
    def __init__(self, hidden, cond_in):
        super(IntegrandNet, self).__init__()
        l1 = [1 + cond_in] + hidden
        l2 = hidden + [1]
        layers = []
        for h1, h2 in zip(l1, l2):
            layers += [nn.Linear(h1, h2), nn.ReLU()]
        layers.pop()
        layers.append(ELUPlus())
        self.net = nn.Sequential(*layers)

    def forward(self, x, h):
        nb_batch, in_d = x.shape
        x = torch.cat((x, h), 1)
        x_he = x.view(nb_batch, -1, in_d).transpose(1, 2).contiguous().view(nb_batch * in_d, -1)
        y = self.net(x_he).view(nb_batch, -1)
        return y


class NeuralIntegral(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, xT, integrand_net, parameters, h, nb_steps):
        if type(integrand_net) is list:
            self.integrand_net = IntegrandNet(integrand_net, cond_size)
        else:
            self.integrand_net = integrand_net
        self.solver = solver
        self.nb_steps = nb_steps
        x0 = torch.zeros(x.shape).to(x.device)
        xT = x
        z0 = h[:, :, 0]
        h = h.permute(0, 2, 1).contiguous().view(x.shape[0], -1)
        if self.solver == "CC":
            z = NeuralIntegral.apply(x0, xT, self.integrand_net, _flatten(self.integrand_net.parameters()),
                                     h, self.nb_steps) + z0
        elif self.solver == "CCParallel":
            z = ParallelNeuralIntegral.apply(x0, xT, self.integrand_net,
                                             _flatten(self.integrand_net.parameters()),
                                             h, self.nb_steps) + z0
        else:
            return None
        return z, self.integrand_net(x, h)

    @staticmethod
    def backward(ctx, grad_output):
        if type(integrand_net) is list:
            self.integrand_net = IntegrandNet(integrand_net, cond_size)
        else:
            self.integrand_net = integrand_net
        self.solver = solver
        self.nb_steps = nb_steps
        x0 = torch.zeros(x.shape).to(x.device)
        xT = x
        z0 = h[:, :, 0]
        h = h.permute(0, 2, 1).contiguous().view(x.shape[0], -1)
        if self.solver == "CC":
            z = NeuralIntegral.apply(x0, xT, self.integrand_net, _flatten(self.integrand_net.parameters()),
                                     h, self.nb_steps) + z0
        elif self.solver == "CCParallel":
            z = ParallelNeuralIntegral.apply(x0, xT, self.integrand_net,
                                             _flatten(self.integrand_net.parameters()),
                                             h, self.nb_steps) + z0
        else:
            return None
        return z, self.integrand_net(x, h)


class ParallelNeuralIntegral(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, xT, integrand_net, parameters, h, nb_steps):
        if type(integrand_net) is list:
            self.integrand_net = IntegrandNet(integrand_net, cond_size)
        else:
            self.integrand_net = integrand_net
        self.solver = solver
        self.nb_steps = nb_steps
        x0 = torch.zeros(x.shape).to(x.device)
        xT = x
        z0 = h[:, :, 0]
        h = h.permute(0, 2, 1).contiguous().view(x.shape[0], -1)
        if self.solver == "CC":
            z = NeuralIntegral.apply(x0, xT, self.integrand_net, _flatten(self.integrand_net.parameters()),
                                     h, self.nb_steps) + z0
        elif self.solver == "CCParallel":
            z = ParallelNeuralIntegral.apply(x0, xT, self.integrand_net,
                                             _flatten(self.integrand_net.parameters()),
                                             h, self.nb_steps) + z0
        else:
            return None
        return z, self.integrand_net(x, h)

    @staticmethod
    def backward(ctx, grad_output):
        if type(integrand_net) is list:
            self.integrand_net = IntegrandNet(integrand_net, cond_size)
        else:
            self.integrand_net = integrand_net
        self.solver = solver
        self.nb_steps = nb_steps
        x0 = torch.zeros(x.shape).to(x.device)
        xT = x
        z0 = h[:, :, 0]
        h = h.permute(0, 2, 1).contiguous().view(x.shape[0], -1)
        if self.solver == "CC":
            z = NeuralIntegral.apply(x0, xT, self.integrand_net, _flatten(self.integrand_net.parameters()),
                                     h, self.nb_steps) + z0
        elif self.solver == "CCParallel":
            z = ParallelNeuralIntegral.apply(x0, xT, self.integrand_net,
                                             _flatten(self.integrand_net.parameters()),
                                             h, self.nb_steps) + z0
        else:
            return None
        return z, self.integrand_net(x, h)


class MonotonicNormalizer(nn.Module):
    def __init__(self, integrand_net, cond_size, nb_steps=20, solver="CC"):
        super(MonotonicNormalizer, self).__init__()
        if type(integrand_net) is list:
            self.integrand_net = IntegrandNet(integrand_net, cond_size)
        else:
            self.integrand_net = integrand_net
        self.solver = {
            "CC": NeuralIntegral,
            "CCParallel": ParallelNeuralIntegral
        }[solver]
        self.nb_steps = nb_steps

    def forward(self, x, h, context=None):
        x0 = torch.zeros(x.shape).to(x.device)
        xT = x
        z0 = h[:, :, 0]
        h = h.permute(0, 2, 1).contiguous().view(x.shape[0], -1)
        z = self.solver.apply(x0, xT, self.integrand_net, torch.flatten(self.integrand_net.parameters()),
                              h, self.nb_steps) + z0
        return z, self.integrand_net(x, h)

    def inverse_transform(self, z, h, context=None):
        # Efficient inversion using bisection method
        x_max = torch.ones_like(z) * 20
        x_min = -torch.ones_like(z) * 20
        z_max, _ = self.forward(x_max, h, context)
        z_min, _ = self.forward(x_min, h, context)
        for i in range(25):
            x_middle = (x_max + x_min) / 2
            z_middle, _ = self.forward(x_middle, h, context)
            mask = (z_middle > z).float()
            x_max = mask * x_middle + (1 - mask) * x_max
            x_min = (1 - mask) * x_middle + mask * x_min
            z_max = mask * z_middle + (1 - mask) * z_max
            z_min = (1 - mask) * z_middle + mask * z_min
        return (x_max + x_min) / 2