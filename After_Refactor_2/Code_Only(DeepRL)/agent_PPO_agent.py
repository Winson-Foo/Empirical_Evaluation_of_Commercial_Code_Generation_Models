from typing import Any, Dict, List, Tuple
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from torch import tensor

from ..network import BaseNetwork
from ..component import Storage
from .BaseAgent import BaseAgent
from ..utils import random_sample, to_np


class PPOAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.total_steps = 0
        self.states = self.config.state_normalizer(self.task.reset())
        self.network = self.config.network_fn()
        self.optim_setup()
        if self.config.shared_repr:
            self.lr_scheduler = LambdaLR(self.opt, lambda step: 1 - step / self.config.max_steps)

    def optim_setup(self):
        self.actor_opt: Optimizer
        self.critic_opt: Optimizer
        if self.config.shared_repr:
            self.opt = self.config.optimizer_fn(self.network.parameters())
        else:
            self.actor_opt = self.config.actor_opt_fn(self.network.actor_params)
            self.critic_opt = self.config.critic_opt_fn(self.network.critic_params)

    def step(self):
        storage = Storage(self.config.rollout_length)
        states = self.states
        for _ in range(self.config.rollout_length):
            prediction = self.network(states)
            action = to_np(prediction['action'])
            next_states, rewards, terminals, info = self.task.step(action)
            rewards = self.config.reward_normalizer(rewards)
            next_states = self.config.state_normalizer(next_states)
            self.record_online_return(info)
            storage.store(states, action, rewards, terminals, prediction)
            states = next_states
            self.total_steps += self.config.num_workers
        self.states = states

        prediction = self.network(states)
        storage.store_prediction(prediction)
        storage.prepare_for_update()

        advantages = tensor([0] * self.config.num_workers).unsqueeze(-1)
        returns = prediction['v'].detach()
        for i in reversed(range(self.config.rollout_length)):
            returns = storage.reward[i] + self.config.discount * storage.mask[i] * returns
            if not self.config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.reward[i] + self.config.discount * storage.mask[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * self.config.gae_tau * self.config.discount * storage.mask[i] + td_error
            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        entries = storage.get_entries(['state', 'action', 'log_pi_a', 'ret', 'advantage'])
        entries.advantage = (entries.advantage - entries.advantage.mean()) / entries.advantage.std()

        if self.config.shared_repr:
            self.lr_scheduler.step(self.total_steps)

        for _ in range(self.config.optimization_epochs):
            sampler = random_sample(range(entries.state.size(0)), self.config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                selected_entries = entries[batch_indices]

                prediction = self.network(selected_entries.state, selected_entries.action)
                ratio = (prediction['log_pi_a'] - selected_entries.log_pi_a).exp()
                obj = ratio * selected_entries.advantage
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip, 1.0 + self.config.ppo_ratio_clip) * selected_entries.advantage
                policy_loss = -torch.min(obj, obj_clipped).mean() - self.config.entropy_weight * prediction['entropy'].mean()

                value_loss = 0.5 * (selected_entries.ret - prediction['v']).pow(2).mean()

                approx_kl = (selected_entries.log_pi_a - prediction['log_pi_a']).mean()
                if self.config.shared_repr:
                    self.opt.zero_grad()
                    (policy_loss + value_loss).backward()
                    clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
                    self.opt.step()
                else:
                    if approx_kl <= 1.5 * self.config.target_kl:
                        self.actor_opt.zero_grad()
                        policy_loss.backward()
                        self.actor_opt.step()
                    self.critic_opt.zero_grad()
                    value_loss.backward()
                    self.critic_opt.step()