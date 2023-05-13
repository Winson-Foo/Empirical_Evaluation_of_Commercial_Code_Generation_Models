import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CategoricalNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, n_atoms):
        super(CategoricalNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim * n_atoms)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(x.size(0), -1, n_atoms)
        prob = F.softmax(x, dim=-1)
        log_prob = F.log_softmax(x, dim=-1)
        return {'prob': prob, 'log_prob': log_prob}