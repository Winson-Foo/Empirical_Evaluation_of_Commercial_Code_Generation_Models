from .network import *
from .storage import *
from .preprocessing import *
from torch.distributions import Categorical

class PPOAgent(BaseAgent):
    def __init__(self, agent_config):
        BaseAgent.__init__(self, agent_config)
        self.agent_config = agent_config
        self.task = agent_config.task_fn()
        self.network = actor_critic_network(agent_config.state_dim, agent_config.action_dim)
        if agent_config.shared_repr:
            self.opt = agent_config.optimizer_fn(self.network.parameters())
        else:
            self.actor_opt = agent_config.actor_opt_fn(self.network.actor_params)
            self.critic_opt = agent_config.critic_opt_fn(self.network.critic_params)
        self.total_steps = 0
        self.states = self.task.reset()
        self.state_normalizer = StateNormalizer(self.states.shape[0])
        self.reward_normalizer = RewardNormalizer()
        self.states = self.state_normalizer(self.states)
        if agent_config.shared_repr:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda step: 1 - step / agent_config.max_steps)

    def step(self):
        config = self.agent_config
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            prediction = self.network(states)
            action = prediction['logits'].detach().numpy()
            action = np.exp(action) / np.sum(np.exp(action))
            action = np.random.choice(range(config.action_dim), p=action)
            next_states, rewards, terminals, info = self.task.step(action)
            reward = self.reward_normalizer(rewards)
            self.reward_normalizer.update(reward)
            next_states = self.state_normalizer(next_states)
            storage.feed({'state': states, 'action': tensor(action).long(),
                           'log_pi_a': prediction['log_pi_a'],
                           'v': prediction['v']})
            storage.feed({'reward': tensor(reward).unsqueeze(-1),
                          'mask': tensor(1 - terminals).unsqueeze(-1)})
            states = next_states
            self.total_steps += config.num_workers
            if self.total_steps % config.target_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

            if self.total_steps % config.log_interval == 0:
                self.logger.log_online_return(info['episode'], info['steps'])

        self.states = states
        prediction = self.network(states)
        storage.feed({'state': states, 'log_pi_a': prediction['log_pi_a'], 'v': prediction['v']})
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

        entries = storage.extract(['state', 'action', 'log_pi_a', 'ret', 'advantage'])
        EntryCLS = entries.__class__
        entries = EntryCLS(*list(map(lambda x: x.detach(), entries)))
        entries.advantage.copy_((entries.advantage - entries.advantage.mean()) / entries.advantage.std())

        if config.shared_repr:
            self.lr_scheduler.step(self.total_steps)

        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(entries.state.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                entry = EntryCLS(*list(map(lambda x: x[batch_indices], entries)))

                prediction = self.network(entry.state, entry.action)
                ratio = (prediction['log_pi_a'] - entry.log_pi_a).exp()
                obj = ratio * entry.advantage
                obj_clipped = ratio.clamp(1.0 - self.agent_config.ppo_ratio_clip,
                                          1.0 + self.agent_config.ppo_ratio_clip) * entry.advantage
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['entropy'].mean()

                value_loss = 0.5 * (entry.ret - prediction['v']).pow(2).mean()

                approx_kl = (entry.log_pi_a - prediction['log_pi_a']).mean()
                if config.shared_repr:
                    self.opt.zero_grad()
                    (policy_loss + value_loss).backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                    self.opt.step()
                else:
                    if approx_kl <= 1.5 * config.target_kl:
                        self.actor_opt.zero_grad()
                        policy_loss.backward()
                        self.actor_opt.step()
                    self.critic_opt.zero_grad()
                    value_loss.backward()
                    self.critic_opt.step()