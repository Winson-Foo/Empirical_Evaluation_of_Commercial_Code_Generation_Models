from typing import NamedTuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..network import Network
from ..component import Tensor, to_np
from .base_agent import BaseAgent
from .storage import TrajectoryStorage


class A2CAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.agent_config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()

    def step(self):
        agent_config = self.agent_config
        trajectory_storage = TrajectoryStorage(agent_config.rollout_length)
        states = self.states

        # Collect trajectories
        for _ in range(agent_config.rollout_length):
            prediction = self.network(agent_config.state_normalizer(states))
            actions = to_np(prediction['action'])
            next_states, rewards, terminals, info = self.task.step(actions)
            self.record_online_return(info)
            rewards = agent_config.reward_normalizer(rewards)

            trajectory_storage.feed(prediction)
            trajectory_storage.feed({
                'reward': Tensor(rewards).unsqueeze(-1),
                'mask': Tensor(np.float32(1 - terminals)).unsqueeze(-1)
            })

            states = next_states
            self.total_steps += agent_config.num_workers

        self.states = states

        # Compute advantages and returns
        prediction = self.network(agent_config.state_normalizer(states))
        trajectory_storage.feed(prediction)
        trajectory_storage.placeholder()

        advantages = Tensor(np.zeros((agent_config.num_workers, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(agent_config.rollout_length)):
            returns = trajectory_storage.reward[i] + agent_config.discount * \
                      trajectory_storage.mask[i] * returns

            if not agent_config.use_gae:
                advantages = returns - trajectory_storage.v[i].detach()
            else:
                temporal_difference_error = trajectory_storage.reward[i] + \
                                            agent_config.discount * trajectory_storage.mask[i] * \
                                            trajectory_storage.v[i + 1] - trajectory_storage.v[i]
                advantages = advantages * agent_config.gae_tau * \
                             agent_config.discount * trajectory_storage.mask[i] + temporal_difference_error

            trajectory_storage.advantage[i] = advantages.detach()
            trajectory_storage.ret[i] = returns.detach()

        # Compute loss
        entries = trajectory_storage.extract(['log_pi_a', 'v', 'ret', 'advantage', 'entropy'])

        policy_loss = -(entries.log_pi_a * entries.advantage).mean()
        value_loss = 0.5 * (entries.ret - entries.v).pow(2).mean()
        entropy_loss = entries.entropy.mean()

        self.optimizer.zero_grad()
        (policy_loss - agent_config.entropy_weight * entropy_loss +
         agent_config.value_loss_weight * value_loss).backward()

        nn.utils.clip_grad_norm_(self.network.parameters(), agent_config.gradient_clip)
        self.optimizer.step()