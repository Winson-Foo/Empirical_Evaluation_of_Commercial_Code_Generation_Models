from ..network import *
from ..component import *
from ..utils import *
from .base_agent import *
from .dqn_agent import *


class QuantileRegressionDqnActor(DqnActor):
    def compute_q_values(self, prediction):
        return to_np(prediction['quantile'].mean(-1))


class QuantileRegressionDqnAgent(DqnAgent):
    def __init__(self, config):
        super().__init__(config)
        self.lock = mp.Lock()
        self.replay = config.replay_fn()
        self.actor = QuantileRegressionDqnActor(config)
        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.batch_indices = range_tensor(config.batch_size)
        self.quantile_weight = 1.0 / config.num_quantiles
        self.cumulative_density = tensor((2 * np.arange(config.num_quantiles) + 1) / (2.0 * config.num_quantiles)).view(1, -1)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)['quantile'].mean(-1)
        action = np.argmax(to_np(q).flatten())
        self.config.state_normalizer.unset_read_only()
        return [action]

    def compute_loss(self, transitions):
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)
        quantiles = self.network(states)['quantile']
        actions = tensor(transitions.action).long()
        quantiles = quantiles[self.batch_indices, actions, :]
        rewards = tensor(transitions.reward).unsqueeze(-1)
        masks = tensor(transitions.mask).unsqueeze(-1)
        quantiles_target = self.compute_target_quantiles(next_states, masks, rewards)
        return self.compute_quantile_regression_loss(quantiles, quantiles_target)

    def compute_target_quantiles(self, next_states, masks, rewards):
        quantiles_next = self.target_network(next_states)['quantile'].detach()
        a_next = torch.argmax(quantiles_next.sum(-1), dim=-1)
        quantiles_next = quantiles_next[self.batch_indices, a_next, :]
        return rewards + self.config.discount ** self.config.n_step * masks * quantiles_next.t().unsqueeze(-1)

    def compute_quantile_regression_loss(self, quantiles, quantiles_target):
        diff = quantiles_target - quantiles
        loss = huber(diff) * (self.cumulative_density - (diff.detach() < 0).float()).abs()
        return loss.sum(-1).mean(1)

    def reduce_loss(self, loss):
        return loss.mean()