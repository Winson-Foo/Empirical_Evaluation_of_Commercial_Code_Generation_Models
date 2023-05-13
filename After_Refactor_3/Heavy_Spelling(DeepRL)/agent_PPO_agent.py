from ..network import *
from ..component import *
from .BaseAgent import *


class PPOAgent(BaseAgent):
    def __init__(self, agent_config):
        super().__init__(agent_config)

        self.config = agent_config
        self.task = agent_config.task_fn()
        self.network = agent_config.network_fn()

        if agent_config.shared_repr:
            self.optimizer = agent_config.optimizer_fn(self.network.parameters())
        else:
            self.actor_optimizer = agent_config.actor_opt_fn(self.network.actor_params)
            self.critic_optimizer = agent_config.critic_opt_fn(self.network.critic_params)

        self.total_steps = 0
        self.states = agent_config.state_normalizer(self.task.reset())

        if agent_config.shared_repr:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda step: 1 - step / agent_config.max_steps
            )

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
            storage.feed({
                'reward': tensor(rewards).unsqueeze(-1),
                'mask': tensor(1 - terminals).unsqueeze(-1),
                'state': tensor(states)
            })

            states = next_states
            self.total_steps += self.config.num_workers

        self.states = states
        prediction = self.network(states)
        storage.feed(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((self.config.num_workers, 1)))
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

        entries = storage.extract(['state', 'action', 'log_pi_a', 'ret', 'advantage'])
        EntryCLS = entries.__class__
        entries = EntryCLS(*list(map(lambda x: x.detach(), entries)))
        entries.advantage.copy_((entries.advantage - entries.advantage.mean()) / entries.advantage.std())

        if self.config.shared_repr:
            self.lr_scheduler.step(self.total_steps)

        for _ in range(self.config.optimization_epochs):
            sampler = random_sample(np.arange(entries.state.size(0)), self.config.mini_batch_size)

            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                entry = EntryCLS(*list(map(lambda x: x[batch_indices], entries)))

                self.optimize_model(entry)

    def optimize_model(self, entry):
        prediction = self.network(entry.state, entry.action)
        ratio = (prediction['log_pi_a'] - entry.log_pi_a).exp()
        obj = ratio * entry.advantage
        obj_clipped = ratio.clamp(
            1.0 - self.config.ppo_ratio_clip,
            1.0 + self.config.ppo_ratio_clip
        ) * entry.advantage
        policy_loss = -torch.min(obj, obj_clipped).mean() - self.config.entropy_weight * prediction['entropy'].mean()
        value_loss = 0.5 * (entry.ret - prediction['v']).pow(2).mean()
        approx_kl = (entry.log_pi_a - prediction['log_pi_a']).mean()

        if self.config.shared_repr:
            self.optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            self.optimizer.step()
        else:
            if approx_kl <= 1.5 * self.config.target_kl:
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()