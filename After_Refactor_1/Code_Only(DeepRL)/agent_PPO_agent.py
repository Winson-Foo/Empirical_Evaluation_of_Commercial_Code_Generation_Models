from typing import List, Dict
from ..network import *
from ..component import *
from .BaseAgent import *


class PPOAgent(BaseAgent):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.actor_opt = config.actor_opt_fn(self.network.actor_params) if not config.shared_repr else None
        self.critic_opt = config.critic_opt_fn(self.network.critic_params) if not config.shared_repr else None
        self.total_steps = 0
        self.states = config.state_normalizer(self.task.reset())
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: 1 - step / config.max_steps) if config.shared_repr else None

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        for _ in range(config.rollout_length):
            prediction = self.network(self.states)
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['action']))
            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            self.record_online_return(info)
            storage.feed({"state": self.states, "action": prediction["action"], 
                          "log_pi_a": prediction["log_pi_a"], "reward": tensor(rewards).unsqueeze(-1), 
                          "mask": tensor(1 - terminals).unsqueeze(-1)})
            self.states = next_states
            self.total_steps += config.num_workers

        prediction = self.network(self.states)
        storage.feed(prediction)
        storage.placeholder()

        returns = prediction["v"].detach()
        advantages = tensor(np.zeros((config.num_workers, 1)))
        for i in reversed(range(config.rollout_length)):
            returns = storage.reward[i] + config.discount * storage.mask[i] * returns
            if config.use_gae:
                td_error = storage.reward[i] + config.discount * storage.mask[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.mask[i] + td_error
            else:
                advantages = returns - storage.v[i].detach()
            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        entries = storage.extract(["state", "action", "log_pi_a", "ret", "advantage"])
        entries.advantage.copy_((entries.advantage - entries.advantage.mean()) / entries.advantage.std())

        for _ in range(config.optimization_epochs):
            for batch_indices in random_sample(np.arange(entries.state.size(0)), config.mini_batch_size):
                batch_indices = tensor(batch_indices).long()
                entry = Storage.from_list([entry_i[batch_indices] for entry_i in entries])

                prediction = self.network(entry.state, entry.action)
                ratio = (prediction['log_pi_a'] - entry.log_pi_a).exp()
                obj = ratio * entry.advantage
                obj_clipped = ratio.clamp(1 - config.ppo_ratio_clip, 1 + config.ppo_ratio_clip) * entry.advantage
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['entropy'].mean()

                value_loss = 0.5 * (entry.ret - prediction['v']).pow(2).mean()

                approx_kl = (entry.log_pi_a - prediction['log_pi_a']).mean()
                if config.shared_repr:
                    self.optimizer.zero_grad()
                    (policy_loss + value_loss).backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                    self.optimizer.step()
                    self.lr_scheduler.step(self.total_steps)
                else:
                    if approx_kl <= 1.5 * config.target_kl:
                        self.actor_opt.zero_grad()
                        policy_loss.backward()
                        self.actor_opt.step()
                    self.critic_opt.zero_grad()
                    value_loss.backward()
                    self.critic_opt.step()