from typing import List
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ..network import Network
from ..component import Config, Task
from ..utils import to_np, epsilon_greedy


class DQNActor:
    def __init__(self, config: Config):
        self.config = config
        self.task: Task = config.task_fn()
        self.network: Network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.start_time = time.time()

    def _get_explore_rate(self, step: int) -> float:
        if step < self.config.exploration_steps:
            return self.config.max_explore_rate - step * \
                   (self.config.max_explore_rate - self.config.min_explore_rate) / self.config.exploration_steps
        return self.config.min_explore_rate

    def _calc_loss(self, batch: List[torch.Tensor]) -> torch.Tensor:
        states, actions, rewards, next_states, is_done = batch

        q_all = self.network(states)
        # Q values corresponding to the actions taken
        q = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_all = self.network(next_states)
            if self.config.double_q:
                next_actions = torch.argmax(q_all, dim=1)
                next_q = next_q_all.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = next_q_all.max(dim=1).values
            expected_q = rewards + (self.config.gamma ** self.config.n_step) * next_q * (1 - is_done)
        return F.smooth_l1_loss(q, expected_q)

    def step(self, step: int) -> tuple:
        s = self.task.reset()
        total_reward = 0
        num_steps = 0
        while True:
            explore_rate = self._get_explore_rate(step)
            if explore_rate > torch.rand(1).item():
                a = self.task.sample_action()
            else:
                with torch.no_grad():
                    self.network.eval()
                    q_all = self.network(torch.FloatTensor([s])).squeeze(0)
                    a = to_np(q_all.argmax(dim=0)).item()
                    self.network.train()

            s_next, r, done, info = self.task.step(a)
            num_steps += 1
            total_reward += r
            exp = [s, a, r, s_next, done]

            self.optimizer.zero_grad()
            loss = self._calc_loss(self.config.transition_cls(*exp))
            loss.backward()
            if self.config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            self.optimizer.step()

            if done:
                break

            s = s_next

        return total_reward, num_steps, step, time.time() - self.start_time, info