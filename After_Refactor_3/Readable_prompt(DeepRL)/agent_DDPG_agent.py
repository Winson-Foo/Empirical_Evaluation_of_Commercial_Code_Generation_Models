# Refactored code:

# Add imports at the top
import numpy as np
import torch

from ..network import ActorCriticNetwork
from ..component import ReplayBuffer
from .base_agent import BaseAgent


class DDPGAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.task = config.task_fn()
        self.network = ActorCriticNetwork(config.state_dim, config.action_dim,
                                          hidden_sizes=config.hidden_sizes)
        self.target_network = ActorCriticNetwork(config.state_dim, config.action_dim,
                                                 hidden_sizes=config.hidden_sizes)
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None

    def soft_update(self, target_network, network):
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network.get_action(state, deterministic=True)
        self.config.state_normalizer.unset_read_only()
        return action.squeeze()

    def step(self):
        config = self.config
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = config.state_normalizer(self.state)

        if self.total_steps < config.warm_up:
            action = self.task.action_space.sample()
        else:
            action = self.network.get_action(self.state, deterministic=False)
            action = np.clip(action.squeeze() + self.random_process.sample(), self.task.action_space.low, self.task.action_space.high)
        next_state, reward, done, info = self.task.step(action)
        next_state = config.state_normalizer(next_state)
        self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)

        self.replay_buffer.add_transition(state=self.state, action=action, reward=reward,
                                           next_state=next_state, mask=1 - done)

        if done:
            self.random_process.reset_states()
            self.state = None
        else:
            self.state = next_state

        self.total_steps += 1

        if self.replay_buffer.size() >= config.batch_size:
            transitions = self.replay_buffer.sample(config.batch_size)
            states = torch.tensor(transitions.state, dtype=torch.float32)
            actions = torch.tensor(transitions.action, dtype=torch.float32)
            rewards = torch.tensor(transitions.reward, dtype=torch.float32).unsqueeze(-1)
            next_states = torch.tensor(transitions.next_state, dtype=torch.float32)
            masks = torch.tensor(transitions.mask, dtype=torch.float32).unsqueeze(-1)

            phi_next = self.target_network.feature(next_states)
            a_next = self.target_network.actor(phi_next)
            q_next = self.target_network.critic(phi_next, a_next)
            q_next = config.discount * masks * q_next
            q_next.add_(rewards)
            q_next = q_next.detach()

            phi = self.network.feature(states)
            current_q = self.network.critic(phi, actions)
            critic_loss = torch.nn.functional.mse_loss(current_q, q_next)

            self.network.critic_opt.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            phi = self.network.feature(states)
            actions_next = self.network.actor(phi)
            policy_loss = -self.network.critic(phi.detach(), actions_next).mean()

            self.network.actor_opt.zero_grad()
            policy_loss.backward()
            self.network.actor_opt.step()

            self.soft_update(self.target_network, self.network)