from typing import List
from torch import optim, tensor
from torch.nn.utils import clip_grad_norm_
from numpy import zeros
from ..network import Network
from ..component import Storage
from .BaseAgent import BaseAgent

class A2CAgent(BaseAgent):
    """
    Advantage Actor-Critic (A2C) agent.
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()

        self.rollout_length = config.rollout_length
        self.num_workers = config.num_workers
        self.state_normalizer = config.state_normalizer
        self.reward_normalizer = config.reward_normalizer
        self.discount = config.discount
        self.use_gae = config.use_gae
        self.gae_tau = config.gae_tau
        self.gradient_clip = config.gradient_clip
        self.value_loss_weight = config.value_loss_weight
        self.entropy_weight = config.entropy_weight

        self.storage = None

    def run_episode(self) -> float:
        """
        Run a single episode and return its total undiscounted reward.

        :return: Total undiscounted reward of the episode
        """
        while not self.task.is_terminal():
            prediction = self.network(self.state_normalizer(self.states))
            next_states, rewards, terminals, info = self.task.step(prediction['action'].detach().numpy())
            self.record_online_return(info)
            rewards = self.reward_normalizer(rewards)
            self.storage.feed(prediction)
            self.storage.feed({
                'reward': tensor(rewards).unsqueeze(-1),
                'mask': tensor(1 - terminals).unsqueeze(-1)
            })
            self.states = next_states
            self.total_steps += self.num_workers
        episode_return = self.get_online_return()
        self.states = self.task.reset()
        return episode_return

    def train(self):
        """
        Train the agent for one iteration over the environment.
        """
        self.storage = Storage(self.rollout_length)

        states = self.states
        for _ in range(self.rollout_length):
            prediction = self.network(self.state_normalizer(states))
            next_states, rewards, terminals, info = self.task.step(prediction['action'].detach().numpy())
            self.record_online_return(info)
            rewards = self.reward_normalizer(rewards)
            self.storage.feed(prediction)
            self.storage.feed({
                'reward': tensor(rewards).unsqueeze(-1),
                'mask': tensor(1 - terminals).unsqueeze(-1)
            })
            states = next_states
            self.total_steps += self.num_workers

        self.states = states

        self.compute_advantages_and_returns()

        self.compute_losses()

    def compute_advantages_and_returns(self):
        """
        Compute the advantages and returns for each time step in the rollout.
        """
        advantages = tensor(zeros((self.num_workers, 1)))
        returns = self.storage.latest_prediction['v'].detach()
        for i in reversed(range(self.rollout_length)):
            returns = self.storage.reward[i] + self.discount * self.storage.mask[i] * returns
            if not self.use_gae:
                advantages = returns - self.storage.v[i].detach()
            else:
                td_error = (self.storage.reward[i] + self.discount * self.storage.mask[i] * self.storage.v[i + 1]
                            - self.storage.v[i])
                advantages = advantages * self.gae_tau * self.discount * self.storage.mask[i] + td_error
            self.storage.advantage[i] = advantages.detach()
            self.storage.ret[i] = returns.detach()

    def compute_losses(self):
        """
        Compute the policy, value, and entropy losses and perform backpropagation.
        """
        entries = self.storage.extract(['log_pi_a', 'v', 'ret', 'advantage', 'entropy'])
        policy_loss = -(entries.log_pi_a * entries.advantage).mean()
        value_loss = 0.5 * (entries.ret - entries.v).pow(2).mean()
        entropy_loss = entries.entropy.mean()
        loss = policy_loss - self.entropy_weight * entropy_loss + self.value_loss_weight * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.network.parameters(), self.gradient_clip)
        self.optimizer.step()