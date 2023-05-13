from ..network import *
from ..component import *
from ..utils import *
from .BaseAgent import *
from .DQN_agent import *


class QuantileRegressionDQNActor(DQNActor):
    def __init__(self, config):
        super().__init__(config)

    def compute_q_values(self, prediction):
        return to_np(prediction['quantile'].mean(-1))


class QuantileRegressionDQNAgent(DQNAgent):
    def __init__(self, config):
        super().__init__(config)
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = QuantileRegressionDQNActor(config)
        self.network = config.network_fn().share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.actor.set_network(self.network)

        self.quantile_weight = 1.0 / self.config.num_quantiles
        self.cumulative_density = tensor(
            [(2 * i + 1) / (2.0 * self.config.num_quantiles) for i in range(self.config.num_quantiles)]).view(1, -1)

    def eval_step(self, state):
        with self.config.state_normalizer.set_read_only():
            q = self.actor.compute_q_values(self.network(self.config.state_normalizer(state)))
            return np.argmax(q)

    def compute_loss(self, transitions):
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)

        quantiles_next = self.target_network(next_states)['quantile'].detach()
        a_next = torch.argmax(quantiles_next.sum(-1), dim=-1)
        quantiles_next = quantiles_next[tensor(range(self.config.batch_size)), a_next]

        rewards = tensor(transitions.reward).unsqueeze(-1)
        masks = tensor(transitions.mask).unsqueeze(-1)
        quantiles_next = rewards + self.config.discount ** self.config.n_step * masks * quantiles_next

        quantiles = self.network(states)['quantile']
        actions = tensor(transitions.action).long()
        quantiles = quantiles[tensor(range(self.config.batch_size)), actions, :]

        quantiles_next = quantiles_next.t().unsqueeze(-1)
        diff = quantiles_next - quantiles
        loss = huber(diff) * (self.cumulative_density - (diff.detach() < 0).float()).abs()
        return loss.sum(-1).mean()

    def reduce_loss(self, loss):
        return loss