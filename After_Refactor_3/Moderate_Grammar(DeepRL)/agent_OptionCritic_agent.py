from contextlib import nullcontext
from typing import List, Tuple
import torch
from torch import nn, optim, tensor
from torch.distributions import Categorical
import numpy as np
from ..component import Storage, to_np
from ..network import Network
from .BaseAgent import BaseAgent


class OptionCriticAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.target_network.load_state_dict(self.network.state_dict())

        self.total_steps = 0
        self.worker_index = tensor(np.arange(config.num_workers)).long()

        self.states = self.config.state_normalizer(self.task.reset())
        self.is_initial_states = tensor(np.ones((config.num_workers))).bool()
        self.prev_options = self.is_initial_states.clone().long()

    def sample_option(self, prediction: dict, epsilon: float, prev_option: torch.Tensor, is_initial_states: torch.Tensor) -> torch.Tensor:
        """
        Samples the new options for each worker using the current state and the previous option as input.

        Args:
            prediction (dict): A dictionary containing the predicted Q and beta values, and the policy pi.
            epsilon (float): Probability of random selection of options.
            prev_option (tensor): Tensor containing the previous option chosen by each worker.
            is_initial_states (tensor): Tensor indicating whether each worker is in an initial state. 

        Returns:
            new_options: Tensor containing the new options selected by each worker.
        """
        q_option, beta = prediction['q'], prediction['beta']
        n_options = q_option.size(1)
        pi_option = (torch.ones_like(q_option) * epsilon / n_options)
        greedy_option = q_option.argmax(dim=-1, keepdim=True)
        pi_option.scatter_(1, greedy_option, (1.0 - epsilon + epsilon / n_options))
        mask = torch.zeros_like(q_option)
        mask[self.worker_index, prev_option] = 1
        pi_hat_option = (1 - beta) * mask + beta * pi_option
        options = Categorical(probs=pi_option).sample()
        options_hat = Categorical(probs=pi_hat_option).sample()
        new_options = torch.where(is_initial_states, options, options_hat)
        return new_options

    def rollout(self) -> Tuple[Storage, List[float]]:
        """
        Executes a rollout of the environment.

        Returns:
            A tuple containing the storage and list of online returns for the current rollout.
        """
        config = self.config
        storage = Storage(config.rollout_length, ['beta', 'option', 'beta_advantage', 'prev_option', 'init_state', 'eps'])

        for _ in range(config.rollout_length):
            prediction = self.network(self.states)
            epsilon = config.random_option_prob(config.num_workers)
            options = self.sample_option(prediction, epsilon, self.prev_options, self.is_initial_states)
            prediction['pi'] = prediction['pi'][self.worker_index, options]
            prediction['log_pi'] = prediction['log_pi'][self.worker_index, options]
            dist = Categorical(probs=prediction['pi'])
            actions = dist.sample()
            entropy = dist.entropy()
            next_states, rewards, terminals, info = self.task.step(to_np(actions))
            self.record_online_return(info)
            next_states = config.state_normalizer(next_states)
            rewards = config.reward_normalizer(rewards)
            storage.feed(prediction)
            storage.feed({'reward': tensor(rewards).unsqueeze(-1),
                          'mask': tensor(1 - terminals).unsqueeze(-1),
                          'option': options.unsqueeze(-1),
                          'prev_option': self.prev_options.unsqueeze(-1),
                          'entropy': entropy.unsqueeze(-1),
                          'action': actions.unsqueeze(-1),
                          'init_state': self.is_initial_states.unsqueeze(-1).float(),
                          'eps': epsilon})
            self.is_initial_states = tensor(terminals).bool()
            self.prev_options = options
            self.states = next_states
            self.total_steps += config.num_workers
            if self.total_steps // config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        with torch.no_grad():
            prediction = self.target_network(self.states)
            storage.placeholder()
            betas = prediction['beta'][self.worker_index, self.prev_options]
            ret = (1 - betas) * prediction['q'][self.worker_index, self.prev_options] + \
                  betas * torch.max(prediction['q'], dim=-1)[0]
            ret = ret.unsqueeze(-1)

        for i in reversed(range(config.rollout_length)):
            ret = storage.reward[i] + config.discount * storage.mask[i] * ret
            adv = ret - storage.q[i].gather(1, storage.option[i])
            storage.ret[i] = ret
            storage.advantage[i] = adv
            v = storage.q[i].max(dim=-1, keepdim=True).values * (1 - storage.eps[i]) + \
                storage.q[i].mean(-1).unsqueeze(-1) * storage.eps[i]
            q = storage.q[i].gather(1, storage.prev_option[i])
            storage.beta_advantage[i] = q - v + config.termination_regularizer

        return storage, self.online_returns

    def update(self, storage: Storage) -> Tuple[float, float, float]:
        """
        Updates the model using the given data.

        Args:
            storage (Storage): Data container that stores the data of the current rollout.

        Returns:
            A tuple containing the losses for the pi, q, and beta networks.
        """
        entries = storage.extract(['q', 'beta', 'log_pi', 'ret', 'advantage', 'beta_advantage',
                                   'entropy', 'option', 'action', 'init_state', 'prev_option'])
        q_loss = (entries.q.gather(1, entries.option) - entries.ret.detach()).pow(2).mul(0.5).mean()
        pi_loss = -(entries.log_pi.gather(1, entries.action) * entries.advantage.detach()) - \
                  self.config.entropy_weight * entries.entropy
        pi_loss = pi_loss.mean()
        beta_loss = torch.where(
            entries.init_state,
            entries.beta.gather(1, entries.prev_option) * entries.beta_advantage.detach(),
            torch.zeros_like(entries.prev_option)
        ).mean()
        loss = pi_loss + q_loss + beta_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        return pi_loss.item(), q_loss.item(), beta_loss.item()

    def learn(self, total_steps: int) -> None:
        """
        Runs the training loop for the given number of total steps.

        Args:
            total_steps (int): Total number of steps to train the agent for.
        """
        while self.total_steps < total_steps:
            with nullcontext() if self.config.disable_training else self.train_logger:
                storage, returns = self.rollout()
                pi_loss, q_loss, beta_loss = self.update(storage)
                self.episode += self.config.num_workers
                if self.episode % self.config.log_interval == 0:
                    self.logger.log({
                        'episode': self.episode,
                        'total_steps': self.total_steps,
                        'return': np.mean(returns),
                        'pi_loss': pi_loss,
                        'q_loss': q_loss,
                        'beta_loss': beta_loss,
                    })