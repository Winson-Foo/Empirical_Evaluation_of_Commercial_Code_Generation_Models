from ..network import *
from ..component import *
from ..utils import *
from .DQN_agent import *


class QuantileRegressionDQNActor(DQNActor):
    def __init__(self, config):
        super().__init__(config)

    def compute_q(self, prediction):
        q_values = prediction['quantile'].mean(-1)
        return to_np(q_values)


class QuantileRegressionDQNAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        
        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        
        self.replay = config.replay_fn()
        self.actor = self.create_actor(self.network, config)
        
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.batch_indices = range_tensor(config.batch_size)
        self.quantile_weight = 1.0 / self.config.num_quantiles
        self.cumulative_density = tensor(
            (2 * np.arange(self.config.num_quantiles) + 1) / (2.0 * self.config.num_quantiles)).view(1, -1)
        
    def create_actor(self, network, config):
        return QuantileRegressionDQNActor(config).set_network(network)
    
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

        quantiles_next = self.target_network(next_states)['quantile'].detach()
        argmax_quantiles_next = torch.argmax(quantiles_next.sum(-1), dim=-1)
        quantiles_next = quantiles_next[self.batch_indices, argmax_quantiles_next, :]

        rewards = tensor(transitions.reward).unsqueeze(-1)
        masks = tensor(transitions.mask).unsqueeze(-1)
        quantiles_next = rewards + self.config.discount ** self.config.n_step * masks * quantiles_next

        quantiles = self.network(states)['quantile']
        actions = tensor(transitions.action).long()
        quantiles = quantiles[self.batch_indices, actions, :]

        quantiles_next = quantiles_next.t().unsqueeze(-1)
        diff = quantiles_next - quantiles
        huber_loss = huber(diff) * (self.cumulative_density - (diff.detach() < 0).float()).abs()
        total_loss = huber_loss.sum(-1).mean(1)
        return total_loss

    def reduce_loss(self, loss):
        return loss.mean()