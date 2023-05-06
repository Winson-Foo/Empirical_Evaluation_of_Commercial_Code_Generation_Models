from ..network import *
from ..component import *
from .BaseAgent import *
import torch.optim.lr_scheduler as lr_scheduler


class PPOAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self._prepare_state(self.task.reset())

    def _prepare_state(self, state):
        state = self.config.state_normalizer(state)
        return state

    def _calculate_advantages(self, storage, prediction):
        returns = prediction['v'].detach()
        advantages = tensor(np.zeros((self.config.num_workers, 1)))
        for i in reversed(range(self.config.rollout_length)):
            returns = storage.reward[i] + self.config.discount * storage.mask[i] * returns
            if not self.config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.reward[i] + self.config.discount * storage.mask[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * self.config.gae_tau * self.config.discount * storage.mask[i] + td_error
            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        entries = storage.extract(['state', 'action', 'log_pi_a', 'ret', 'advantage'])
        EntryCLS = entries.__class__
        entries = EntryCLS(*list(map(lambda x: x.detach(), entries)))
        entries.advantage.copy_((entries.advantage - entries.advantage.mean()) / entries.advantage.std())

        return entries

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            prediction = self.network(states)
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['action']))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            next_states = self._prepare_state(next_states)
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

        entries = self._calculate_advantages(storage, prediction)

        if config.shared_repr:
            self.optimizer_scheduler.step(self.total_steps)

        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(entries.state.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                entry = EntryCLS(*list(map(lambda x: x[batch_indices], entries)))

                prediction = self.network(entry.state, entry.action)
                ratio = (prediction['log_pi_a'] - entry.log_pi_a).exp()
                obj = ratio * entry.advantage
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * entry.advantage
                policy_loss = -torch.min(obj, obj_clipped).mean() - self.config.entropy_weight * prediction['entropy'].mean()

                value_loss = 0.5 * (entry.ret - prediction['v']).pow(2).mean()

                approx_kl = (entry.log_pi_a - prediction['log_pi_a']).mean()
                if config.shared_repr:
                    self.optimizer.zero_grad()
                    (policy_loss + value_loss).backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                    self.optimizer.step()
                else:
                    if approx_kl <= 1.5 * config.target_kl:
                        self.actor_optimizer.zero_grad()
                        policy_loss.backward()
                        self.actor_optimizer.step()
                    self.critic_optimizer.zero_grad()
                    value_loss.backward()
                    self.critic_optimizer.step()