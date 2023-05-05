class PPOAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.storage = Storage(config.rollout_length)

        if config.shared_repr:
            self.optimizer = config.optimizer_fn(self.network.parameters())
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda step: 1 - step / config.max_steps)
        else:
            self.actor_optimizer = config.actor_opt_fn(self.network.actor_params)
            self.critic_optimizer = config.critic_opt_fn(self.network.critic_params)

        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)

    def step(self):
        config = self.config
        self.storage.reset()
        states = self.states

        for _ in range(config.rollout_length):
            prediction = self.network(states)
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['action']))
            self.record_online_return(info)

            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            self.storage.feed(prediction)
            self.storage.feed({
                'reward': tensor(rewards).unsqueeze(-1),
                'mask': tensor(1 - terminals).unsqueeze(-1),
                'state': tensor(states)
            })
            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction = self.network(states)
        self.storage.feed(prediction)
        self.storage.placeholder()

        self._compute_advantages_returns()

        self.storage.normalize_advantages()

        self._optimize_policy_value()

    def _compute_advantages_returns(self):
        config = self.config
        storage = self.storage

        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = storage.prediction['v'].detach()

        for i in reversed(range(config.rollout_length)):
            returns = storage.reward[i] + config.discount * storage.mask[i] * returns

            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = (
                    storage.reward[i] +
                    config.discount * storage.mask[i] * storage.v[i + 1] -
                    storage.v[i]
                )
                advantages = advantages * config.gae_tau * config.discount * storage.mask[i] + td_error

            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()

    def _optimize_policy_value(self):
        config = self.config
        storage = self.storage

        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(storage.size), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                entries = storage[batch_indices]

                policy_loss = self._compute_policy_loss(entries)
                value_loss = self._compute_value_loss(entries)

                if config.shared_repr:
                    self.optimizer.zero_grad()
                    (policy_loss + value_loss).backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                    self.optimizer.step()
                    self.lr_scheduler.step(self.total_steps)
                else:
                    if self._kl_divergence(entries) <= 1.5 * config.target_kl:
                        self.actor_optimizer.zero_grad()
                        policy_loss.backward()
                        self.actor_optimizer.step()
                    self.critic_optimizer.zero_grad()
                    value_loss.backward()
                    self.critic_optimizer.step()

    def _compute_policy_loss(self, entries):
        prediction = self.network(entries.state, entries.action)
        ratio = (prediction['log_pi_a'] - entries.log_pi_a).exp()
        obj = ratio * entries.advantage
        obj_clipped = ratio.clamp(
            1.0 - self.config.ppo_ratio_clip,
            1.0 + self.config.ppo_ratio_clip
        ) * entries.advantage
        policy_loss = -torch.min(obj, obj_clipped).mean() - self.config.entropy_weight * prediction['entropy'].mean()

        return policy_loss

    def _compute_value_loss(self, entries):
        prediction = self.network(entries.state, entries.action)
        value_loss = 0.5 * (entries.ret - prediction['v']).pow(2).mean()

        return value_loss

    def _kl_divergence(self, entries):
        prediction = self.network(entries.state, entries.action)
        return (entries.log_pi_a - prediction['log_pi_a']).mean()