from ..network import *
from ..component import *
from .BaseAgent import *

class Network:
    def __init__(self, config):
        self.actor = config.network_fn()['actor']
        self.critic = config.network_fn()['critic']
        self.shared_repr = config.shared_repr

        if self.shared_repr:
            self.parameters = self.actor.parameters()
        else:
            self.actor_params = list(self.actor.parameters())
            self.critic_params = list(self.critic.parameters())

    def forward(self, states, actions=None):
        if self.shared_repr:
            policy = self.actor(states)
            value = self.critic(states)
        else:
            policy = self.actor(states)
            if actions is None:
                value = self.critic(states)
            else:
                value = self.critic(states, actions)

        return {'action': policy.sample(),
                'log_pi_a': policy.log_prob(policy.sample()),
                'v': value,
                'entropy': policy.entropy()}


class Optimizer:
    def __init__(self, config, network):
        self.config = config
        self.network = network

        if self.network.shared_repr:
            self.optimizer = config.optimizer_fn(self.network.parameters())
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: 1 - step / config.max_steps)
        else:
            self.actor_optimizer = config.actor_opt_fn(self.network.actor_params)
            self.critic_optimizer = config.critic_opt_fn(self.network.critic_params)

    def step(self, policy_loss, value_loss):
        if self.network.shared_repr:
            self.optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            self.optimizer.step()
            self.scheduler.step()
        else:
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            if self.config.shared_critic:
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                self.critic_optimizer.step()


class Loader:
    def __init__(self, buffer, mini_batch_size):
        self.buffer = buffer
        self.mini_batch_size = mini_batch_size

    def __iter__(self):
        indices = np.arange(self.buffer.size)
        np.random.shuffle(indices)

        for start in range(0, self.buffer.size, self.mini_batch_size):
            batch_indices = indices[start:start + self.mini_batch_size]
            batch = self.buffer.sample(batch_indices)
            yield {
                'state': batch.state,
                'action': batch.action,
                'log_pi_a': batch.log_pi_a,
                'ret': batch.ret,
                'advantage': batch.advantage,
                'mask': batch.mask
            }


class PPOAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.task = config.task_fn()
        self.buffer = Storage(config.rollout_length)
        self.network = Network(config)
        self.optimizer = Optimizer(config, self.network)
        self.lr_scheduler = self.optimizer.scheduler if self.network.shared_repr else None
        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)

    def step(self):
        for _ in range(self.config.rollout_length):
            prediction = self.network.forward(self.states)
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['action']))
            self.record_online_return(info)
            rewards = self.config.reward_normalizer(rewards)
            next_states = self.config.state_normalizer(next_states)
            self.buffer.feed(prediction)
            self.buffer.feed({'reward': tensor(rewards).unsqueeze(-1),
                               'mask': tensor(1 - terminals).unsqueeze(-1),
                               'state': tensor(self.states)})
            self.states = next_states
            self.total_steps += self.config.num_workers

        self.buffer.placeholder()

        advantages = tensor(np.zeros((self.config.num_workers, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(self.config.rollout_length)):
            returns = self.buffer.reward[i] + self.config.discount * self.buffer.mask[i] * returns
            if not self.config.use_gae:
                advantages = returns - self.buffer.v[i].detach()
            else:
                td_error = self.buffer.reward[i] + self.config.discount * self.buffer.mask[i] * self.buffer.v[i + 1] - self.buffer.v[i]
                advantages = advantages * self.config.gae_tau * self.config.discount * self.buffer.mask[i] + td_error
            self.buffer.advantage[i] = advantages.detach()
            self.buffer.ret[i] = returns.detach()

        self.buffer.advantage.copy_((self.buffer.advantage - self.buffer.advantage.mean()) / self.buffer.advantage.std())

        self.train()

    def train(self):
        for _ in range(self.config.optimization_epochs):
            for batch in Loader(self.buffer, self.config.mini_batch_size):
                entry = Namespace(**batch)

                prediction = self.network.forward(entry.state, entry.action)
                ratio = (prediction['log_pi_a'] - entry.log_pi_a).exp()
                obj = ratio * entry.advantage
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip, 1.0 + self.config.ppo_ratio_clip) * entry.advantage
                policy_loss = -torch.min(obj, obj_clipped).mean() - self.config.entropy_weight * prediction['entropy'].mean()

                value_loss = 0.5 * (entry.ret - prediction['v']).pow(2).mean()

                self.optimizer.step(policy_loss, value_loss)

        self.buffer.reset()