from typing import List, Dict

from torch import Tensor
from torch.nn.functional import mse_loss

from ..network import Network
from ..component import Storage
from ..task import Task
from ..utils import tensor, to_np, epsilon_greedy


class Normalizer:
    def __init__(self, state_dims: int, reward_dims: int):
        self.state_dims = state_dims
        self.reward_dims = reward_dims

    def normalize_state(self, state: List[float]) -> Tensor:
        return tensor(state).view(1, self.state_dims)

    def normalize_reward(self, reward: float) -> float:
        return reward


class NStepDQNAgent:
    def __init__(self, config):
        self.config = config
        self.task: Task = config.task_fn()
        self.network: Network = config.network_fn()
        self.target_network: Network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())

        self.total_steps = 0
        self.states = self.task.reset()

        self.normalizer = Normalizer(self.config.state_dims, self.config.reward_dims)

    def train(self):
        storage = Storage(self.config.rollout_length)

        states = self.states
        for _ in range(self.config.rollout_length):
            q_values = self.network(self.normalizer.normalize_state(states))['q']

            epsilon = self.config.random_action_prob(self.config.num_workers)
            actions = epsilon_greedy(epsilon, to_np(q_values))

            next_states, rewards, terminals, info = self.task.step(actions)
            self.record_online_return(info)
            rewards = self.normalizer.normalize_reward(rewards)

            storage.feed({'q': q_values,
                          'action': tensor(actions).unsqueeze(-1).long(),
                          'reward': tensor(rewards).unsqueeze(-1),
                          'mask': tensor(1 - terminals).unsqueeze(-1)})

            states = next_states

            self.total_steps += self.config.num_workers
            if self.total_steps // self.config.num_workers % self.config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        self.states = states

        storage.placeholder()

        q_next = self.target_network(self.normalizer.normalize_state(states))['q'].detach()
        q_next_max, _ = q_next.max(dim=1, keepdim=True)

        for i in reversed(range(self.config.rollout_length)):
            target = storage.reward[i] + self.config.discount * storage.mask[i] * q_next_max
            storage.ret[i] = target

        entries = storage.extract(['q', 'action', 'ret'])
        loss = mse_loss(entries.q.gather(1, entries.action), entries.ret)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()

    def step(self):
        self.train()