from typing import List
import torch
import torch.multiprocessing as mp
from torch.optim import Optimizer
from torch.nn.utils import range as range_tensor
from .BaseAgent import BaseAgent
from .DQN_agent import DQNActor, DQNAgent
from ..component import HuberLoss
from ..network import Network
from ..utils import tensor, to_np


class QuantileRegressionDQNActor(DQNActor):
    def __init__(self, config):
        super().__init__(config)

    def compute_q_values(self, prediction) -> torch.Tensor:
        q_values = prediction['quantile'].mean(dim=-1)
        return to_np(q_values)


class QuantileRegressionDQNAgent(DQNAgent):
    def __init__(self, config):
        super().__init__(config)
        self.replay_buffer = config.replay_fn()
        self.actor = QuantileRegressionDQNActor(config)
        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.actor.set_network(self.network)
        self.total_steps = 0
        self.batch_indices = range_tensor(config.batch_size)
        self.quantile_weight = 1.0 / self.config.num_quantiles
        self.cumulative_density = tensor(
            (2 * torch.arange(self.config.num_quantiles) + 1) /
            (2.0 * self.config.num_quantiles)).view(1, -1)

    def eval_step(self, state: torch.Tensor) -> List[int]:
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)['quantile'].mean(dim=-1)
        action = torch.argmax(to_np(q).flatten())
        self.config.state_normalizer.unset_read_only()
        return [int(action)]

    def compute_loss(self, transitions):
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)

        quantiles_next = self.target_network(next_states)['quantile'].detach()
        a_next = torch.argmax(quantiles_next.sum(-1), dim=-1)
        quantiles_next = quantiles_next[self.batch_indices, a_next, :]

        rewards = tensor(transitions.reward).unsqueeze(dim=-1)
        masks = tensor(transitions.mask).unsqueeze(dim=-1)
        quantiles_next = rewards + self.config.discount ** self.config.n_step * masks * quantiles_next

        quantiles = self.network(states)['quantile']
        actions = tensor(transitions.action).long()
        quantiles = quantiles[self.batch_indices, actions, :]

        quantiles_next = quantiles_next.t().unsqueeze(dim=-1)
        diff = quantiles_next - quantiles
        huber_loss = HuberLoss()
        loss = huber_loss(diff) * (self.cumulative_density - (diff.detach() < 0).float()).abs()
        return loss.sum(dim=-1).mean(dim=1)

    def reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss.mean()