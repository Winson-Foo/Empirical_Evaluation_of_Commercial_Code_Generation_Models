from contextlib import contextmanager
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from typing import List, Tuple
import numpy as np
import torch
from .BaseAgent import BaseAgent
from ..network import Network
from ..component import Storage


class A2CAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states

        # Collect experiences
        for _ in range(config.rollout_length):
            with torch.no_grad():
                prediction = self.network(config.state_normalizer(states))
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['action']))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            storage.feed(prediction)
            storage.feed({'reward': tensor(rewards).unsqueeze(-1),
                          'mask': tensor(1 - terminals).unsqueeze(-1)})
            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        with torch.no_grad():
            prediction = self.network(config.state_normalizer(states))
        storage.feed(prediction)
        storage.placeholder()

        # Compute advantages and returns
        advantages, returns = compute_advantages_and_returns(config, storage)

        # Compute losses
        policy_loss, value_loss, entropy_loss = compute_losses(config, storage)

        # Update network
        update_network(config, self.optimizer, self.network, policy_loss, value_loss, entropy_loss)

    @staticmethod
    @contextmanager
    def _managed_optimizer(optimizer: Optimizer):
        optimizer.zero_grad()
        yield
        clip_grad_norm_(optimizer.parameters(), config.gradient_clip)
        optimizer.step()

    @staticmethod
    @contextmanager
    def _managed_storage(config):
        storage = Storage(config.rollout_length)
        yield storage
        del storage

    def compute_advantages_and_returns(self, config, storage):
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.reward[i] + config.discount * storage.mask[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.reward[i] + config.discount * storage.mask[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.mask[i] + td_error
            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()
        return advantages, returns

    def compute_losses(config, storage) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entries = storage.extract(['log_pi_a', 'v', 'ret', 'advantage', 'entropy'])
        policy_loss = -(entries.log_pi_a * entries.advantage).mean()
        value_loss = 0.5 * (entries.ret - entries.v).pow(2).mean()
        entropy_loss = entries.entropy.mean()
        return policy_loss, value_loss, entropy_loss

    def update_network(config, optimizer, network, policy_loss, value_loss, entropy_loss):
        with self._managed_optimizer(optimizer):
            (policy_loss - config.entropy_weight * entropy_loss +
             config.value_loss_weight * value_loss).backward()
        nn.utils.clip_grad_norm_(network.parameters(), config.gradient_clip)