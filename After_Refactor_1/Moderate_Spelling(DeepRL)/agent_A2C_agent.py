from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from ..network import Network
from ..component import Storage
from .BaseAgent import BaseAgent
from ..utils import to_np, tensor


class A2CAgent(BaseAgent):
    """
    A2C Agent that uses Advantage Actor-Critic algorithm to improve RL training
    """
    def __init__(self, config):
        """
        Initializes the A2C Agent.

        :param config: configuration file with necessary parameters for the agent
        """
        super().__init__(config)
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()

    def train_step(self, states: np.ndarray, 
                   storage: Storage) -> Dict[str, torch.Tensor]:
        """
        Runs a single training step on the current batch.

        :param states: current states input
        :param storage: history of previous states, actions, and rewards
        :return: dictionary of losses: policy, value, and entropy
        """
        config = self.config
        policy_loss, value_loss, entropy_loss = 0, 0, 0

        # gather experience
        for _ in range(config.rollout_length):
            prediction = self.network(config.state_normalizer(states))
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['action']))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            storage.feed(prediction)
            storage.feed({'reward': tensor(rewards).unsqueeze(-1),
                           'mask': tensor(1 - terminals).unsqueeze(-1)})
            states = next_states
            self.total_steps += config.num_workers

        # calculate advantages and returns
        prediction = self.network(config.state_normalizer(states))
        storage.feed(prediction)
        storage.placeholder()
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

        # calculate losses and backpropagate
        entries = storage.extract(['log_pi_a', 'v', 'ret', 'advantage', 'entropy'])
        policy_loss = -(entries.log_pi_a * entries.advantage).mean()
        value_loss = 0.5 * (entries.ret - entries.v).pow(2).mean()
        entropy_loss = entries.entropy.mean()

        self.optimizer.zero_grad()
        (policy_loss - config.entropy_weight * entropy_loss +
         config.value_loss_weight * value_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

        # return loss dictionary
        return {'policy': policy_loss, 'value': value_loss, 'entropy': entropy_loss}

    def step(self):
        """
        Steps the environment and updates the agent's network.
        """
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states

        # train on multiple environments
        for _ in range(config.num_workers):
            train_losses = self.train_step(states, storage)
        self.states = states

        return train_losses