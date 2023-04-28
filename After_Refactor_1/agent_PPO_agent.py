from typing import NamedTuple
import torch
import torch.nn.functional as F
from torch import nn, tensor
import numpy as np

from ..network import *
from ..component import *
from .BaseAgent import *


class PPOAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.task = config.task_fn()
        self.network = config.network_fn()
        if config.shared_repr:
            self.optimizer = config.optimizer_fn(self.network.parameters())
        else:
            self.actor_opt = config.actor_opt_fn(self.network.actor_params)
            self.critic_opt = config.critic_opt_fn(self.network.critic_params)
        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        if config.shared_repr:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: 1 - step / config.max_steps)

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            prediction = self.network(states)
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['action']))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            storage.feed(prediction)
            storage.feed({'reward': tensor(rewards).unsqueeze(-1),
                           'mask': tensor(1 - terminals).unsqueeze(-1),
                           'state': tensor(states)})
            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction = self.network(states)
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

        class Entry(NamedTuple):
            state: torch.Tensor
            action: torch.Tensor
            log_pi_a: torch.Tensor
            ret: torch.Tensor
            advantage: torch.Tensor

        entries = storage.extract(['state', 'action', 'log_pi_a', 'ret', 'advantage'])
        entries = Entry(*list(map(lambda x: x.detach(), entries)))
        entries.advantage.copy_((entries.advantage - entries.advantage.mean()) / entries.advantage.std())

        if config.shared_repr:
            self.lr_scheduler.step(self.total_steps)

        def actor_optimization():
            sampler = random_sample(np.arange(entries.state.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                entry = Entry(*list(map(lambda x: x[batch_indices], entries)))

                prediction = self.network(entry.state, entry.action)
                ratio = (prediction['log_pi_a'] - entry.log_pi_a).exp()
                obj = ratio * entry.advantage
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * entry.advantage
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['entropy'].mean()

                yield policy_loss

        def critic_optimization():
            sampler = random_sample(np.arange(entries.state.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                entry = Entry(*list(map(lambda x: x[batch_indices], entries)))

                prediction = self.network(entry.state)
                value_loss = 0.5 * (entry.ret - prediction['v']).pow(2).mean()

                yield value_loss

        if config.shared_repr:
            for loss in actor_optimization():
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.optimizer.step()
        else:
            if approx_kl <= 1.5 * config.target_kl:
                for loss in actor_optimization():
                    self.actor_opt.zero_grad()
                    loss.backward()
                    self.actor_opt.step()

            for loss in critic_optimization():
                self.critic_opt.zero_grad()
                loss.backward()
                self.critic_opt.step()