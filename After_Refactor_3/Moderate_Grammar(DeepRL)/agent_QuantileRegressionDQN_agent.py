from ..network import *
from ..component import *
from ..utils import *
from .BaseAgent import *
from .DQN_agent import *


class DQNActor(BaseActor):
    def __init__(self, agent_config):
        super().__init__(agent_config)
        self.network = None

    def set_network(self, network):
        self.network = network

    def compute_q(self, obs):
        q_values = self.network(obs)['logits']
        return to_np(q_values).flatten()


class QuantileRegressionDQNActor(DQNActor):
    def __init__(self, agent_config):
        super().__init__(agent_config)

    def compute_q(self, obs):
        prediction = self.network(obs)
        q_values = prediction['quantile'].mean(-1)
        return to_np(q_values).flatten()


class QuantileRegressionDQNAgent(DQNAgent):
    def __init__(self, agent_config):
        super().__init__(agent_config)

        self.replay = agent_config.replay_fn()
        self.actor = QuantileRegressionDQNActor(agent_config)
        self.actor.set_network(self.network)

    def eval_step(self, obs):
        self.config.state_normalizer.set_read_only()
        obs = self.config.state_normalizer(obs)
        q = self.actor.compute_q(obs)
        action = np.argmax(q)
        self.config.state_normalizer.unset_read_only()
        return [action]

    def compute_loss(self, batch):
        states, actions, rewards, next_states, masks = batch
        states = self.config.state_normalizer(states)
        next_states = self.config.state_normalizer(next_states)

        quantiles_next = self.target_network(next_states)['quantile'].detach()
        a_next = torch.argmax(quantiles_next.sum(-1), dim=-1)
        quantiles_next = quantiles_next[range_tensor(self.config.batch_size), a_next, :]

        rewards = tensor(rewards).unsqueeze(-1)
        masks = tensor(masks).unsqueeze(-1)
        quantiles_next = rewards + self.config.discount ** self.config.n_step * masks * quantiles_next

        quantiles = self.network(states)['quantile']
        quantiles = quantiles[range_tensor(self.config.batch_size), actions, :]

        quantiles_next = quantiles_next.t().unsqueeze(-1)
        diff = quantiles_next - quantiles
        cumulative_density = tensor((2 * np.arange(self.config.num_quantiles) + 1) / (2.0 * self.config.num_quantiles)).view(1, -1)
        loss = huber(diff) * (cumulative_density - (diff.detach() < 0).float()).abs()
        return loss.sum(-1).mean()

    def reduce_loss(self, loss):
        return loss.mean()