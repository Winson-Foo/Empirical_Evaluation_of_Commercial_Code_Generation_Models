import torch.nn as nn

class Network(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(Network, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

    def forward(self, x):
        raise NotImplementedError

class ActorCriticNetwork(Network):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256)):
        super(ActorCriticNetwork, self).__init__(input_shape=[state_dim], num_actions=action_dim)

        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
        )

        self.actor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim)
        )

        self.critic = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )

    def forward(self, state, action=None):
        features = self.features(state)
        logits = self.actor(features)
        values = self.critic(features)

        dist = Categorical(logits=logits)
        log_prob = None
        entropy = None
        if action is not None:
            log_prob = dist.log_prob(action)
            entropy = dist.entropy().mean()
        return {'logits': logits, 'log_pi_a': log_prob, 'v': values, 'entropy': entropy}