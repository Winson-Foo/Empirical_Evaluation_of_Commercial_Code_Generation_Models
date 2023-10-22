from ..network import *
from ..component import *
from .BaseAgent import *


class PPOAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.actor_opt = config.actor_opt_fn(self.network.actor_params)
        self.critic_opt = config.critic_opt_fn(self.network.critic_params)
        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.actor_opt, lambda step: 1 - step / config.max_steps)
        self.config = config

    def step(self):
        storage = Storage(self.config.rollout_length)
        states = self.states
        for _ in range(self.config.rollout_length):
            prediction = self.network(states)
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['action']))
            self.record_online_return(info)
            rewards = self.config.reward_normalizer(rewards)
            next_states = self.config.state_normalizer(next_states)
            storage.feed(prediction)
            storage.feed({'reward': tensor(rewards).unsqueeze(-1),
                           'mask': tensor(1 - terminals).unsqueeze(-1),
                           'state': tensor(states)})
            states = next_states
            self.total_steps += self.config.num_workers

        self.states = states
        prediction = self.network(states)
        storage.feed(prediction)
        storage.placeholder()
        self.calculate_advantages(storage)

        entries = storage.extract(['state', 'action', 'log_pi_a', 'ret', 'advantage'])
        self.normalize_advantages(entries)
        self.optimize_networks(entries)

    def calculate_advantages(self, storage):
        advantages = tensor(np.zeros((self.config.num_workers, 1)))
        returns = storage.prediction['v'].detach()
        for i in reversed(range(self.config.rollout_length)):
            returns = storage.reward[i] + self.config.discount * storage.mask[i] * returns
            if not self.config.use_gae:
                advantages = returns - storage.prediction['v'][i].detach()
            else:
                td_error = storage.reward[i] + self.config.discount * storage.mask[i] * storage.prediction['v'][i + 1] - storage.prediction['v'][i]
                advantages = advantages * self.config.gae_tau * self.config.discount * storage.mask[i] + td_error
            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()

    def normalize_advantages(self, entries):
        entries.advantage.copy_((entries.advantage - entries.advantage.mean()) / entries.advantage.std())

    def optimize_networks(self, entries):
        for _ in range(self.config.optimization_epochs):
            sampler = random_sample(np.arange(entries.state.size(0)), self.config.mini_batch_size)
            for batch_indices in sampler:
                self.optimize(actor=self.config.shared_repr, entries=entries, batch_indices=batch_indices)

    def optimize(self, actor, entries, batch_indices):
        entry = entries[batch_indices]
        prediction = self.network(entry.state, entry.action)
        ratio = (prediction['log_pi_a'] - entry.log_pi_a).exp()
        obj = ratio * entry.advantage
        obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                  1.0 + self.config.ppo_ratio_clip) * entry.advantage
        policy_loss = -torch.min(obj, obj_clipped).mean() - self.config.entropy_weight * prediction['entropy'].mean()

        value_loss = 0.5 * (entry.ret - prediction['v']).pow(2).mean()

        approx_kl = (entry.log_pi_a - prediction['log_pi_a']).mean()
        if actor:
            self.actor_opt.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.network.actor_params, self.config.gradient_clip)
            self.actor_opt.step()
            self.lr_scheduler.step(self.total_steps)
        else:
            if approx_kl <= 1.5 * self.config.target_kl:
                self.actor_opt.zero_grad()
                policy_loss.backward()
                self.actor_opt.step()
            self.critic_opt.zero_grad()
            value_loss.backward()
            self.critic_opt.step()