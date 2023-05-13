# File: distributions.py

import torch

class DiagonalNormal(torch.distributions.Normal):
    def __init__(self, mean, std):
        super().__init__(mean, std)

    def log_prob(self, action):
        return super().log_prob(action).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1).unsqueeze(-1)

    def cdf(self, action):
        return super().cdf(action).prod(-1).unsqueeze(-1)

class BatchCategorical(torch.distributions.Categorical):
    def __init__(self, logits):
        super().__init__(logits=logits.view(-1, logits.size(-1)))

    def log_prob(self, action):
        log_pi = super().log_prob(action.view(-1))
        log_pi = log_pi.view(action.size()[:-1] + (-1,))
        return log_pi

    def entropy(self):
        ent = super().entropy()
        ent = ent.view(self.pre_shape + (-1,))
        return ent

    def sample(self, sample_shape=torch.Size([])):
        ret = super().sample(sample_shape)
        ret = ret.view(sample_shape + self.pre_shape + (-1,))
        return ret