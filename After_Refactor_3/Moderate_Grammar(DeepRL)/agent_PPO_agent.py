import torch.nn.utils as nn_utils

from ..network import *
from ..component import *
from .BaseAgent import *

class PPOAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        if config.shared_repr:
            self.optimizer = config.optimizer_fn(self.network.parameters())
            self.scheduler = config.scheduler_fn(self.optimizer, config.max_steps)
        else:
            self.actor_opt = config.actor_opt_fn(self.network.actor_params)
            self.critic_opt = config.critic_opt_fn(self.network.critic_params)
        
        self.total_steps = 0
        self.states = config.state_normalizer(self.task.reset())
        self.online_returns = []
        

    def rollout(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for i in range(config.rollout_length):
            dist = self.network(states)
            next_states, rewards, terminals, info = self.task.step(to_np(dist['action']))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            storage.states[i] = states
            storage.actions[i] = dist['action']
            storage.log_probs[i] = dist['log_prob']
            storage.rewards[i] = tensor(rewards).unsqueeze(-1)
            storage.masks[i] = tensor(1 - terminals).unsqueeze(-1)
            states = next_states
            self.total_steps += config.num_workers
        self.states = states
        storage.states[-1] = states
        dist = self.network(states)
        storage.values.copy_(dist['value'])
        
        returns = tensor(np.zeros((config.num_workers, 1)))
        advantages = tensor(np.zeros((config.num_workers, 1)))
        for i in reversed(range(config.rollout_length)):
            returns = storage.rewards[i] + config.gamma * storage.masks[i] * returns
            if not config.use_gae:
                td_error = returns - storage.values[i]
                advantages = td_error.detach()
            else:
                td_error = storage.rewards[i] + config.gamma * storage.masks[i] * storage.values[i + 1] - storage.values[i]
                advantages = advantages * config.gae_tau * config.gamma * storage.masks[i] + td_error.detach()
            storage.advantages[i] = advantages
            storage.returns[i] = returns
        return storage

    def optimize(self, storage):
        config = self.config
        policy_loss = 0
        value_loss = 0
        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(config.rollout_length), config.batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                rewards = storage.rewards[batch_indices]
                masks = storage.masks[batch_indices]
                states = storage.states[batch_indices]
                actions = storage.actions[batch_indices]
                old_log_probs = storage.log_probs[batch_indices].detach()
                returns = storage.returns[batch_indices]
                advantages = storage.advantages[batch_indices]
                dist = self.network(states, actions)
                log_probs = dist['log_prob']
                entropy = dist['entropy'].mean()

                ratio = (log_probs - old_log_probs).exp()
                obj = ratio * advantages
                obj_clipped = ratio.clamp(1.0 - config.ppo_ratio_clip, 1.0 + config.ppo_ratio_clip) * advantages
                policy_loss += -(torch.min(obj, obj_clipped).mean() + entropy * config.entropy_weight)

                value_loss += 0.5 * (dist['value'] - returns).pow(2).mean()

                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                nn_utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.optimizer.step()

        self.scheduler.step()
        
    def step(self):
        storage = self.rollout()
        self.optimize(storage)
