import torch.nn as nn
from typing import Tuple
from torch.optim import Optimizer
from ..network import BaseNetwork
from ..component import Storage, to_np, tensor
from .BaseAgent import BaseAgent


class A2CAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.total_steps = 0
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.states = self.task.reset()

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states

        for _ in range(config.rollout_length):
            prediction = self.network(config.state_normalizer(states))
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['action']))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)

            storage.feed(prediction)
            storage.feed({"reward": tensor(rewards).unsqueeze(-1),
                          "mask": tensor(1 - terminals).unsqueeze(-1)})
            
            states = next_states
            self.total_steps += config.num_workers
        
        self._update_states(states, storage)
        self._compute_losses(storage)
        self._optimize()

    def _update_states(self, states, storage):
        prediction = self.network(self.config.state_normalizer(states))
        storage.feed(prediction)
        storage.placeholder()

        returns = prediction["v"]
        advantages = tensor(torch.zeros((self.config.num_workers, 1)))
        for i in reversed(range(self.config.rollout_length)):
            returns = storage.reward[i] + self.config.discount * storage.mask[i] * returns
            if not self.config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.reward[i] + self.config.discount * storage.mask[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * self.config.gae_tau * self.config.discount * storage.mask[i] + td_error
            
            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        self.states = states

    def _compute_losses(self, storage):
        entries = storage.extract(['log_pi_a', 'v', 'ret', 'advantage', 'entropy'])
        policy_loss = -(entries.log_pi_a * entries.advantage).mean()
        value_loss = 0.5 * (entries.ret - entries.v).pow(2).mean()
        entropy_loss = entries.entropy.mean()
        
        self.losses = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item()
        }

        self.total_loss = (policy_loss - self.config.entropy_weight * entropy_loss +
                           self.config.value_loss_weight * value_loss)

    def _optimize(self):
        self.optimizer.zero_grad()
        self.total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        
    def reset(self):
        self.states = self.task.reset()
        self.total_steps = 0

    def best_action(self, state) -> Tuple:
        with torch.no_grad():
            state = tensor(state).unsqueeze(0)
            prediction = self.network(self.config.state_normalizer(state))
            action = to_np(prediction["action"])
        
        return action[0]