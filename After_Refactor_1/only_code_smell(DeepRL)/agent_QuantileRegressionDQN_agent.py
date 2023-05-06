from network import *
from component import *
from utils import *
from DQN_agent import *


class QuantileRegressionDQNActor(DQNActor):
    def __init__(self, config):
        super().__init__(config)

    def compute_q_values(self, prediction):
        q_values = prediction['quantile'].mean(dim=-1)
        return to_np(q_values)


class QuantileRegressionDQNAgent(DQNAgent):
    def __init__(self, config):
        super().__init__(config)
        
        self.replay = config.replay_fn()
        self.actor = QuantileRegressionDQNActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)

        self.total_steps = 0
        self.batch_indices = torch.arange(config.batch_size)

        self.num_quantiles = config.num_quantiles
        self.quantile_weight = 1.0 / self.num_quantiles
        self.cumulative_density = torch.tensor(
            (2 * np.arange(self.num_quantiles) + 1) / (2.0 * self.num_quantiles)).view(1, -1)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q_values = self.network(state)['quantile'].mean(dim=-1)
        action = np.argmax(to_np(q_values).flatten())
        self.config.state_normalizer.unset_read_only()
        return [action]

    def compute_loss(self, transitions):
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)

        quantiles_next = self.target_network(next_states)['quantile'].detach()
        a_next = torch.argmax(quantiles_next.sum(dim=-1), dim=-1)
        quantiles_next = quantiles_next[self.batch_indices, a_next, :]

        rewards = torch.tensor(transitions.reward).unsqueeze(-1)
        masks = torch.tensor(transitions.mask).unsqueeze(-1)
        quantiles_next = rewards + self.config.discount ** self.config.n_step * masks * quantiles_next

        quantiles = self.network(states)['quantile']
        actions = torch.tensor(transitions.action).long()
        quantiles = quantiles[self.batch_indices, actions, :]

        quantiles_next = quantiles_next.t().unsqueeze(-1)
        diff = quantiles_next - quantiles
        loss = huber(diff) * (self.cumulative_density - (diff.detach() < 0).float()).abs()
        return loss.sum(dim=-1).mean(dim=1)

    def reduce_loss(self, loss):
        return loss.mean()