from ..network import *
from ..component import *
from .BaseAgent import *

class PPOAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda step: 1 - step / config.max_steps
        )
        self.rollout_length = config.rollout_length
        self.total_steps = 0
        self.states = config.state_normalizer(self.task.reset())

    def step(self):
        storage = Storage(self.rollout_length)
        for _ in range(self.rollout_length):
            prediction = self.network(self.states)
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['action']))
            self.record_online_return(info)
            rewards = self.config.reward_normalizer(rewards)
            next_states = self.config.state_normalizer(next_states)
            storage.feed(prediction)
            storage.feed({
                'reward': tensor(rewards).unsqueeze(-1),
                'mask': tensor(1 - terminals).unsqueeze(-1),
                'state': tensor(self.states)
            })
            self.states = next_states
            self.total_steps += self.config.num_workers

        prediction = self.network(self.states)
        storage.feed(prediction)
        storage.placeholder()

        advantages, returns = self._compute_advantages_and_returns(storage)

        entries = storage.extract(['state', 'action', 'log_pi_a', 'ret', 'advantage'])
        entries = self._normalize_advantages(entries)

        self._update_network(entries)

    def _compute_advantages_and_returns(self, storage):
        advantages = tensor(np.zeros((self.config.num_workers, 1)))
        returns = storage.prediction['v'].detach()
        for i in reversed(range(self.rollout_length)):
            returns = storage.reward[i] + self.config.discount * storage.mask[i] * returns
            if not self.config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.reward[i] + self.config.discount * storage.mask[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * self.config.gae_tau * self.config.discount * storage.mask[i] + td_error
            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()
        return advantages, returns

    def _normalize_advantages(self, entries):
        EntryCLS = entries.__class__
        entries.advantage.copy_((entries.advantage - entries.advantage.mean()) / entries.advantage.std())
        return entries

    def _compute_losses(self, entries, prediction):
        ratio = (prediction['log_pi_a'] - entries.log_pi_a).exp()
        obj = ratio * entries.advantage
        obj_clipped = ratio.clamp(
            1.0 - self.config.ppo_ratio_clip, 1.0 + self.config.ppo_ratio_clip) * entries.advantage
        policy_loss = -torch.min(obj, obj_clipped).mean() - self.config.entropy_weight * prediction['entropy'].mean()
        value_loss = 0.5 * (entries.ret - prediction['v']).pow(2).mean()
        approx_kl = (entries.log_pi_a - prediction['log_pi_a']).mean()
        return policy_loss, value_loss, approx_kl

    def _update_network(self, entries):
        for _ in range(self.config.optimization_epochs):
            sampler = random_sample(np.arange(entries.state.size(0)), self.config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                entry = entries.__class__(*list(map(lambda x: x[batch_indices], entries)))
                prediction = self.network(entry.state, entry.action)
                policy_loss, value_loss, approx_kl = self._compute_losses(entry, prediction)
                if self.config.shared_repr:
                    self.optimizer.zero_grad()
                    (policy_loss + value_loss).backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
                    self.optimizer.step()
                else:
                    if approx_kl <= 1.5 * self.config.target_kl:
                        self.actor_opt.zero_grad()
                        policy_loss.backward()
                        self.actor_opt.step()
                    self.critic_opt.zero_grad()
                    value_loss.backward()
                    self.critic_opt.step()