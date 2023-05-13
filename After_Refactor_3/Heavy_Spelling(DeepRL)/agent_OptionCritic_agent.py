from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
from ..component import Storage
from .base_agent import BaseAgent


class OptionCritic:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.task = config.task_fn()
        self.q_network = config.q_network_fn()
        self.target_q_network = config.q_network_fn()
        self.policy_network = config.policy_network_fn()
        self.optimizer = config.optimizer_fn(self.q_network.parameters())
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def train(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
              next_states: torch.Tensor, terminals: torch.Tensor, is_initial_states: torch.Tensor) -> float:
        config = self.config
        with torch.no_grad():
            target_prediction = self.target_q_network(next_states).detach()
            target_q_value, _ = torch.max(target_prediction['q'], dim=-1, keepdim=True)
            target_beta = target_prediction['beta']
            target_rewards_to_go = (1 - target_beta) * target_q_value \
                                   + target_beta * target_prediction['v'].unsqueeze(-1)

        prediction = self.q_network(states)
        q_value = prediction['q']
        beta = prediction['beta']
        pi = self.policy_network(states)
        log_pi = torch.log(torch.clamp(pi, min=config.epsilon))
        entropy = -(pi * log_pi).sum(dim=-1)

        selected_q_value = torch.gather(q_value, dim=-1, index=actions)
        advantage = selected_q_value - prediction['v']
        beta_advantage = selected_q_value - q_value.max(dim=-1, keepdim=True)[0]

        q_loss = 0.5 * (selected_q_value - rewards + config.discount
                        * (1 - terminals) * target_rewards_to_go).pow(2).mean()
        pi_loss = -(log_pi.gather(dim=-1, index=actions) * advantage.detach()).mean()
        entropy_loss = -config.entropy_weight * entropy.mean()
        beta_loss = (beta.gather(dim=-1, index=is_initial_states.long())
                     * beta_advantage.detach() * (1 - is_initial_states.float())).mean()
        loss = q_loss + pi_loss + entropy_loss + beta_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), config.gradient_clip)
        self.optimizer.step()
        return loss.item()


class OptionCriticAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.config = config
        self.option_critic = OptionCritic(config)
        self.storage = Storage(config.rollout_length,
                               ['state', 'action', 'reward', 'next_state', 'terminal', 'is_initial_state'])
        self.worker_indices = torch.arange(config.num_workers).long()
        self.states = config.state_normalizer(self.task.reset())
        self.is_initial_states = torch.ones((config.num_workers), dtype=torch.uint8)
        self.prev_options = self.is_initial_states.long()

    def sample_option(self, prediction: Dict[str, torch.Tensor], epsilon: float,
                      prev_option: torch.Tensor, is_initial_states: torch.Tensor) -> torch.Tensor:
        q_option = prediction['q']
        pi_option = torch.zeros_like(q_option).add(epsilon / q_option.size(1))
        greedy_option = q_option.argmax(dim=-1, keepdim=True)
        prob = 1 - epsilon + epsilon / q_option.size(1)
        prob = torch.zeros_like(pi_option).add(prob)
        pi_option.scatter_(1, greedy_option, prob)

        mask = torch.zeros_like(q_option)
        mask[self.worker_indices, prev_option] = 1
        beta = prediction['beta']
        pi_hat_option = (1 - beta) * mask + beta * pi_option

        dist = distributions.Categorical(probs=pi_option)
        options = dist.sample()
        dist = distributions.Categorical(probs=pi_hat_option)
        options_hat = dist.sample()

        options = torch.where(is_initial_states, options, options_hat)
        return options

    def step(self) -> None:
        config = self.config
        for _ in range(config.rollout_length):
            prediction = self.option_critic.q_network(self.states)
            epsilon = config.random_option_prob(config.num_workers)
            options = self.sample_option(prediction, epsilon, self.prev_options, self.is_initial_states)
            dist = distributions.Categorical(probs=self.policy_network(self.states))
            actions = dist.sample()
            entropy = dist.entropy()

            next_states, rewards, terminals, _ = self.task.step(to_np(actions))
            self.record_online_return(_)
            next_states = config.state_normalizer(next_states)
            rewards = config.reward_normalizer(rewards)
            self.storage.feed({'state': self.states,
                                'action': actions.unsqueeze(-1),
                                'reward': tensor(rewards).unsqueeze(-1),
                                'next_state': next_states,
                                'terminal': tensor(terminals).unsqueeze(-1),
                                'is_initial_state': self.is_initial_states.unsqueeze(-1)})
            self.is_initial_states[:] = tensor(terminals).byte()
            self.prev_options[:] = options
            self.states[:] = next_states

            self.option_critic.train(self.storage.state,
                                      self.storage.action,
                                      self.storage.reward,
                                      self.storage.next_state,
                                      self.storage.terminal,
                                      self.storage.is_initial_state)

            self.total_steps += config.num_workers
            if self.total_steps % config.target_network_update_freq == 0:
                self.option_critic.target_q_network.load_state_dict(self.option_critic.q_network.state_dict())

        self.storage.clear_buffer()