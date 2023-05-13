from ..component import ReplayBuffer
from ..network import TwinDelayedDeepDeterministicPolicyGradientNet
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F

class TwinDelayedDeepDeterministicPolicyGradientAgent:
    def __init__(self, config):
        self.config = config
        self.task = config.task_fn()
        self.network = TwinDelayedDeepDeterministicPolicyGradientNet(config.state_dim, config.action_dim, config.hidden_dim)
        self.target_network = TwinDelayedDeepDeterministicPolicyGradientNet(config.state_dim, config.action_dim, config.hidden_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay_buffer = ReplayBuffer(config.replay_size)
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None

    def step(self, config):
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = config.state_normalizer(self.state)

        if self.total_steps < config.warm_up:
            action = [self.task.action_space.sample()]
        else:
            action = self.network(self.state)
            action = to_np(action)
            action += self.random_process.sample()
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        next_state, reward, done, info = self.task.step(action)
        next_state = config.state_normalizer(next_state)
        self.record_online_return(info)
        reward = config.reward_normalizer(reward)

        self.replay_buffer.feed(dict(
            state=self.state,
            action=action,
            reward=reward,
            next_state=next_state,
            mask=1-np.asarray(done, dtype=np.int32),
        ))

        if done[0]:
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        if self.total_steps >= config.warm_up:
            transitions = self.replay_buffer.sample()
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

            if self.total_steps % config.td3_delay == 0:
                action = self.network(states)
                policy_loss = -self.network.q(states, action)[0].mean()

                self.network.zero_grad()
                policy_loss.backward()
                self.network.actor_opt.step()

                self.soft_update(self.target_network, self.network, config.target_network_mix)

    @staticmethod
    def soft_update(target_net, src_net, tau: float) -> None:
        with torch.no_grad():
            for target_param, param in zip(target_net.parameters(), src_net.parameters()):
                target_param.detach_()
                target_param.copy_(target_param * (1.0 - tau) + param * tau)

    def eval_step(self, state):
        state = self.config.state_normalizer(state)
        action = self.network(state)
        return to_np(action)