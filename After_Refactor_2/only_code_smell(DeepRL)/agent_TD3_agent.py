from ..network import *
from ..component import *
from .BaseAgent import *


class TD3Agent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None

    def normalize_states(self, states):
        self.config.state_normalizer.set_read_only()
        normalized_states = self.config.state_normalizer(states)
        self.config.state_normalizer.unset_read_only()
        return normalized_states

    def eval_step(self, state):
        state = self.normalize_states(state)
        action = self.network(state)
        return to_np(action)

    def step(self):
        config = self.config
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.normalize_states(self.task.reset())

        if self.total_steps < config.warm_up:
            action = [self.task.action_space.sample()]
        else:
            action = self.network(self.state)
            action = to_np(action)
            action += self.random_process.sample()
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        next_state, reward, done, info = self.task.step(action)
        next_state = self.normalize_states(next_state)
        self.record_online_return(info)
        reward = config.reward_normalizer(reward)

        self.replay.feed(dict(
            state=self.state,
            action=action,
            reward=reward,
            next_state=next_state,
            mask=1 - done.astype(int),
        ))

        if done[0]:
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        if self.total_steps >= config.warm_up:
            transitions = self.replay.sample()
            states = tensor(transitions.state)
            actions = tensor(transitions.action)
            rewards = tensor(transitions.reward).unsqueeze(-1)
            next_states = tensor(transitions.next_state)
            mask = tensor(transitions.mask).unsqueeze(-1)

            next_actions = self.target_network(next_states)
            noise = torch.randn_like(next_actions).mul(config.td3_noise)
            noise = noise.clamp(-config.td3_noise_clip, config.td3_noise_clip)

            min_action = float(self.task.action_space.low[0])
            max_action = float(self.task.action_space.high[0])
            next_actions = (next_actions + noise).clamp(min_action, max_action)

            q1_next, q2_next = self.target_network.q(next_states, next_actions)
            target = rewards + config.discount * mask * torch.min(q1_next, q2_next)
            target = target.detach()

            q1, q2 = self.network.q(states, actions)
            critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            if self.total_steps % config.td3_delay == 0:
                current_actions = self.network(states)
                actor_loss = -self.network.q(states, current_actions)[0].mean()

                self.network.zero_grad()
                actor_loss.backward()
                self.network.actor_opt.step()

                self.soft_update(self.target_network, self.network)

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)