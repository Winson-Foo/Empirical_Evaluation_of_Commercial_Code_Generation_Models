#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import to_np, tensor
from ..component import huber
from .BaseAgent import BaseAgent
from .DQN_agent import DQNActor, DQNAgent


class QuantileRegressionDQNActor(DQNActor):
    def __init__(self, config):
        DQNActor.__init__(self, config)

    def compute_q(self, prediction):
        q_values = prediction['quantile'].mean(-1)
        return to_np(q_values)


class QuantileRegressionDQNAgent(DQNAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config.lock = mp.Lock()
        self.replay = config.replay_fn()
        self.actor = QuantileRegressionDQNActor(config)
        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.actor.set_network(self.network)
        self.total_steps = 0
        self.batch_indices = tensor(range(config.batch_size))
        self.quantile_weight = 1.0 / self.config.num_quantiles
        self.cumulative_density = tensor(
            (2 * np.arange(self.config.num_quantiles) + 1) / (2.0 * self.config.num_quantiles)).view(1, -1)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)['quantile'].mean(-1)
        action = np.argmax(to_np(q).flatten())
        self.config.state_normalizer.unset_read_only()
        return [action]

    def compute_quantiles(self, states):
        return self.network(states)['quantile']

    def compute_next_quantiles(self, transitions):
        next_states = self.config.state_normalizer(transitions.next_state)
        quantiles_next = self.target_network(next_states)['quantile']
        a_next = torch.argmax(quantiles_next.sum(-1), dim=-1)
        quantiles_next = quantiles_next[self.batch_indices, a_next, :]
        return quantiles_next

    def compute_loss(self, transitions):
        states = self.config.state_normalizer(transitions.state)
        quantiles = self.compute_quantiles(states)
        actions = tensor(transitions.action).long()
        quantiles = quantiles[self.batch_indices, actions, :]
        quantiles_next = self.compute_next_quantiles(transitions)
        rewards = tensor(transitions.reward).unsqueeze(-1)
        masks = tensor(transitions.mask).unsqueeze(-1)
        quantiles_next = rewards + self.config.discount ** self.config.n_step * masks * quantiles_next
        quantiles_next = quantiles_next.t().unsqueeze(-1)
        diff = quantiles_next - quantiles
        loss = huber(diff) * (self.cumulative_density - (diff.detach() < 0).float()).abs()
        return loss.sum(-1).mean(1)

    def reduce_loss(self, loss):
        return loss.mean()