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
        else:
            self.actor_optimizer = config.actor_opt_fn(self.network.actor_params)
            self.critic_optimizer = config.critic_opt_fn(self.network.critic_params)

        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)

        if config.shared_repr:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda step: 1 - step / config.max_steps
            )

    def step(self):
        config = self.config
        rollout_buffer = RolloutBuffer(config.rollout_length)
        states = self.states

        for _ in range(config.rollout_length):
            prediction = self.network(states)
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['action']))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            rollout_buffer.add(states, to_np(prediction['action']), rewards, 1 - terminals, prediction)
            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction = self.network(states)
        rollout_buffer.add_last(prediction)

        advantages = torch.zeros((config.num_workers, 1))
        returns = prediction['v'].detach()

        for i in reversed(range(config.rollout_length)):
            returns = rollout_buffer.rewards[i] + config.discount * rollout_buffer.masks[i] * returns
            if not config.use_gae:
                advantages = returns - rollout_buffer.values[i].detach()
            else:
                td_error = rollout_buffer.rewards[i] + config.discount * rollout_buffer.masks[i] * rollout_buffer.values[i + 1] - rollout_buffer.values[i]
                advantages = advantages * config.gae_tau * config.discount * rollout_buffer.masks[i] + td_error

            rollout_buffer.advantages[i] = advantages.detach()
            rollout_buffer.returns[i] = returns.detach()

        entries = rollout_buffer.get_entries()
        entries.advantages.copy_((entries.advantages - entries.advantages.mean()) / entries.advantages.std())

        if config.shared_repr:
            self.lr_scheduler.step(self.total_steps)

        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(entries.states.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                entry = Entry(*list(map(lambda x: x[batch_indices], entries)))

                prediction = self.network(entry.states, entry.actions)
                ratio = (prediction['log_pi_a'] - entry.log_pi_a).exp()
                obj = ratio * entry.advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip, 1.0 + self.config.ppo_ratio_clip) * entry.advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['entropy'].mean()

                value_loss = 0.5 * (entry.returns - prediction['v']).pow(2).mean()

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


class RolloutBuffer:
    def __init__(self, rollout_length):
        self.rollout_length = rollout_length
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.values = []

    def add(self, state, action, reward, mask, prediction):
        for key, value in prediction.items():
            prediction[key] = value.detach()
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.masks.append(mask)
        self.values.append(prediction['v'])

    def add_last(self, prediction):
        for key, value in prediction.items():
            prediction[key] = value.detach()
        self.values.append(prediction['v'])

    def get_entries(self):
        return Entry(
            tensor(self.states),
            tensor(self.actions),
            tensor(self.rewards).unsqueeze(-1),
            tensor(self.masks).unsqueeze(-1),
            tensor(self.values).unsqueeze(-1),
            tensor(np.zeros((len(self.states), 1))),
            tensor(np.zeros((len(self.states), 1))),
        )


class Entry:
    def __init__(self, states, actions, rewards, masks, values, log_pi_a, returns, advantages):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.values = values
        self.log_pi_a = log_pi_a
        self.returns = returns
        self.advantages = advantages