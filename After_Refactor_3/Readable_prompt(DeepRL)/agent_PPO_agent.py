# Refactored code

# improve the maintainability and readability of the original code and refactored it as follows:

import torch.optim as optim


class PPOAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config) # use super() to simplify the __init__() method and replace self.__init__()
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = getattr(optim, config.optimizer)(self.network.parameters(), **config.optimizer_params) 
        self.policy_opt = getattr(optim, config.actor_optimizer)(self.network.actor_params, **config.actor_optimizer_params)
        self.value_opt = getattr(optim, config.critic_optimizer)(self.network.critic_params, **config.critic_optimizer_params)  
        self.total_steps = 0
        self.states = config.state_normalizer(self.task.reset())
        self.lr_scheduler = None
        if config.shared_repr:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: 1 - step / config.max_steps)

    def step(self):
        config = self.config
        rollout = RolloutStorage(config.rollout_length, self.config.num_workers, self.states)

        for _ in range(config.rollout_length):
            with torch.no_grad():
                prediction = self.network(rollout.get_last_states())
                rollout.add_outputs(prediction)

            actions = to_np(prediction['a'])
            next_states, rewards, terminals, info = self.task.step(actions)
            rewards = config.reward_normalizer(rewards)
            rollout.add_returns(rewards, terminals, config.discount)
            rollout.add_next_states(config.state_normalizer(next_states))

            self.record_online_return(info)
            self.total_steps += self.config.num_workers

        with torch.no_grad():
            prediction = self.network(rollout.get_last_states())
            rollout.add_outputs(prediction)
            rollout.compute_returns(prediction['v'], config.use_gae, config.gae_tau, config.gamma)

        advantages = rollout.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # This renormalization step is added here to normalize the advantages of different states 
        # and penalize particular state values so that the agent can learn to learn more generalized policies. 

        for _ in range(config.optimization_epochs):
            sampler = BatchSampler(SubsetRandomSampler(random_indexes(self.config.num_workers, rollout.length)),
                                   config.mini_batch_size * self.config.num_workers, drop_last=True)

            for indices in sampler:
                batch = rollout.extract(indices)

                dist = self.network(batch.states).dist
                log_probs = dist.log_prob(batch.actions).sum(-1, keepdim=True)

                ratio = torch.exp(log_probs - batch.log_probs)
                surr1 = ratio * batch.advantages
                surr2 = torch.clamp(ratio, 1.0 - config.ppo_ratio_clip, 1.0 + config.ppo_ratio_clip) * batch.advantages
                loss = -torch.min(surr1, surr2).mean() - config.entropy_weight * dist.entropy().mean()

                value_loss = (batch.returns - self.network(batch.states).v).pow(2).mean()
                approx_kl = (log_probs - batch.log_probs).mean()

                if config.shared_repr:
                  self.optimizer.zero_grad()
                  (loss + config.value_loss_weight * value_loss).backward()
                  nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                  self.optimizer.step()
                  if self.lr_scheduler is not None:
                    self.lr_scheduler.step(self.total_steps)
                else:
                  self.policy_opt.zero_grad()
                  loss.backward(retain_graph=True)
                  nn.utils.clip_grad_norm_(self.network.actor_params, config.gradient_clip)
                  self.policy_opt.step()

                  self.value_opt.zero_grad()
                  value_loss.backward()
                  nn.utils.clip_grad_norm_(self.network.critic_params, config.gradient_clip)
                  self.value_opt.step()

        self.states = config.state_normalizer(next_states)