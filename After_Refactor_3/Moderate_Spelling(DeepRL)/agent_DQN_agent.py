import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, obs):
        return {"q": self.net(obs)}