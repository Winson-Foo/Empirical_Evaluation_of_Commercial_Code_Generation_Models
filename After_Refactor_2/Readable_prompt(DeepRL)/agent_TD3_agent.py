from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision


class TD3Agent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.init_agent(config)

    def init_agent(self, config):
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return action.cpu().data.numpy().flatten()

    def step(self):
        config = self.config
        if self.state is None:
            self.reset_random_process_state()
            self.reset_current_state()
            return

        if self.total_steps < config.warm_up:
            action = [self.task.action_space.sample()]
        else:
            action = self.network(self.state)
            action = self.add_noise_to_action(action)
        action = self.clip_action_to_space_bounds(action)
        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)

        self.replay.feed(dict(
            state=self.state,
            action=action,
            reward=reward,
            next_state=next_state,
            mask=1-np.asarray(done, dtype=np.int32),
        ))

        self.reset_random_process_state_if_done(done)
        self.set_current_state_to_next_state(next_state)
        self.increment_total_steps()

        if self.has_enough_steps_to_update_agent_weights():
            self.update_agent_weights(config)

    def reset_random_process_state_if_done(self, done):
        if done[0]:
            self.reset_random_process_state()

    def reset_random_process_state(self):
        self.random_process.reset_states()

    def reset_current_state(self):
        self.random_process.reset_states()
        self.state = self.task.reset()
        self.state = self.config.state_normalizer(self.state)

    def set_current_state_to_next_state(self, next_state):
        self.state = next_state

    def add_noise_to_action(self, action):
        action = to_np(action)
        action += self.random_process.sample()
        return action

    def clip_action_to_space_bounds(self, action):
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        return action

    def increment_total_steps(self):
        self.total_steps += 1

    def has_enough_steps_to_update_agent_weights(self):
        return self.total_steps >= self.config.warm_up

    def update_agent_weights(self, config):
        transitions = self.replay.sample()
        states = tensor(transitions.state)
        actions = tensor(transitions.action)
        rewards = tensor(transitions.reward).unsqueeze(-1)
        next_states = tensor(transitions.next_state)
        mask = tensor(transitions.mask).unsqueeze(-1)

        a_next = self.target_network(next_states)
        noise = torch.randn_like(a_next).mul(config.td3_noise)
        noise = noise.clamp(-config.td3_noise_clip, config.td3_noise_clip)

        min_a = float(self.task.action_space.low[0])
        max_a = float(self.task.action_space.high[0])
        a_next = (a_next + noise).clamp(min_a, max_a)

        q_1, q_2 = self.target_network.q(next_states, a_next)
        target = rewards + config.discount * mask * torch.min(q_1, q_2)
        target = target.detach()

        q_1, q_2 = self.network.q(states, actions)
        critic_loss = F.mse_loss(q_1, target) + F.mse_loss(q_2, target)

        self.network.zero_grad()
        critic_loss.backward()
        self.network.critic_opt.step()

        if self.total_steps % config.td3_delay:
            action = self.network(states)
            policy_loss = -self.network.q(states, action)[0].mean()

            self.network.zero_grad()
            policy_loss.backward()
            self.network.actor_opt.step()

            self.soft_update(self.target_network, self.network)