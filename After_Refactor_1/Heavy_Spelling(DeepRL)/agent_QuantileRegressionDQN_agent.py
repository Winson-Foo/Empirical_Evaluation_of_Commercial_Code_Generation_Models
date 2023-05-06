from typing import List
from multiprocessing import Lock
from torch import Tensor
from ..component import ReplayBuffer
from ..utils import range_tensor, to_np
from .BaseAgent import BaseAgent
from .DQN_agent import DQNActor, DQNAgent


class QuantileRegressionDQNActor(DQNActor):
    def compute_q_values(self, prediction: dict) -> Tensor:
        q_values = prediction['quantile'].mean(-1)
        return to_np(q_values)


class QuantileRegressionDQNAgent(DQNAgent):
    def __init__(self, agent_config: dict):
        super().__init__(agent_config)
        self.agent_config = agent_config
        self.agent_config.lock = Lock()

        self.replay_buffer = ReplayBuffer(agent_config)
        self.actor_network = QuantileRegressionDQNActor(agent_config)

        self.network = agent_config.network_fn()
        self.network.share_memory()
        self.target_network = agent_config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = agent_config.optimizer_fn(self.network.parameters())

        self.actor_network.set_network(self.network)

        self.total_steps = 0
        self.batch_indices = range_tensor(agent_config.batch_size)

        self.QUANTILE_WEIGHT = 1.0 / self.agent_config.num_quantiles
        self.CUMULATIVE_DENSITY = Tensor(
            (2 * np.arange(self.agent_config.num_quantiles) + 1) /
            (2.0 * self.agent_config.num_quantiles)).view(1, -1)

    def eval_step(self, state: List) -> List:
        self.agent_config.state_normalizer.set_read_only()
        state = self.agent_config.state_normalizer(state)
        q = self.network(state)['quantile'].mean(-1)
        action = np.argmax(to_np(q).flatten())
        self.agent_config.state_normalizer.unset_read_only()
        return [action]

    def compute_loss(self, transitions: dict) -> Tensor:
        states = self.agent_config.state_normalizer(transitions.state)
        next_states = self.agent_config.state_normalizer(transitions.next_state)

        quantiles_next = self.target_network(next_states)['quantile'].detach()
        a_next = torch.argmax(quantiles_next.sum(-1), dim=-1)
        quantiles_next = quantiles_next[self.batch_indices, a_next, :]

        rewards = Tensor(transitions.reward).unsqueeze(-1)
        masks = Tensor(transitions.mask).unsqueeze(-1)
        quantiles_next = rewards + self.agent_config.discount ** self.agent_config.n_step * masks * quantiles_next

        quantiles = self.network(states)['quantile']
        actions = Tensor(transitions.action).long()
        quantiles = quantiles[self.batch_indices, actions, :]

        quantiles_next = quantiles_next.t().unsqueeze(-1)
        diff = quantiles_next - quantiles
        loss = huber(diff) * (self.CUMULATIVE_DENSITY - (diff.detach() < 0).float()).abs()
        return loss.sum(-1).mean(1)

    def reduce_loss(self, loss: Tensor) -> Tensor:
        return loss.mean()