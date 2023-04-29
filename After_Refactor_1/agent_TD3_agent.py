from typing import Dict
import torch
import torch.nn.functional as F
import numpy as np

from ..network import Network
from ..component import ReplayBuffer, RandomProcess
from .BaseAgent import BaseAgent


class TD3Agent(BaseAgent):
    """
    TD3 (Twin Delayed DDPG) algorithm implementation.
    """

    def __init__(self, config):
        """
        Initialize the TD3Agent instance.
        :param config: (dict) configuration parameters.
        """
        super(TD3Agent, self).__init__(config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay_buffer = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None

    def soft_update(self, target: Network, src: Network):
        """
        Update target network parameters based on source network parameters.
        :param target: (Network) target network.
        :param src: (Network) source network.
        """
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

    def eval_step(self, state):
        """
        Evaluate the next action based on the current state.
        :param state: (ndarray) current state of the environment.
        :return: (ndarray) next action to take.
        """
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return action.cpu().detach().numpy()

    def step(self):
        """
        Execute one step of the TD3 algorithm.
        """
        config = self.config
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = config.state_normalizer(self.state)

        if self.total_steps < config.warm_up:
            action = [self.task.action_space.sample()]
        else:
            action = self.network(self.state)
            action = action.cpu().detach().numpy()
            action += self.random_process.sample()
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)

        self.replay_buffer.add(dict(
            state=self.state,
            action=action,
            reward=reward,
            next_state=next_state,
            mask=(1 - np.asarray(done, dtype=np.int32)),
        ))

        if done[0]:
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        if self.total_steps >= config.warm_up:
            transitions = self.replay_buffer.sample()
            states = torch.tensor(transitions.state, dtype=torch.float32, device=config.device)
            actions = torch.tensor(transitions.action, dtype=torch.float32, device=config.device)
            rewards = torch.tensor(transitions.reward, dtype=torch.float32, device=config.device).unsqueeze(-1)
            next_states = torch.tensor(transitions.next_state, dtype=torch.float32, device=config.device)
            mask = torch.tensor(transitions.mask, dtype=torch.float32, device=config.device).unsqueeze(-1)

            # Compute next action with added noise
            a_next = self.target_network(next_states)
            noise = torch.randn_like(a_next).mul(config.td3_noise)
            noise = noise.clamp(-config.td3_noise_clip, config.td3_noise_clip)
            min_a = float(self.task.action_space.low[0])
            max_a = float(self.task.action_space.high[0])
            a_next = (a_next + noise).clamp(min_a, max_a)

            # Compute target Q value
            q_1, q_2 = self.target_network.q(next_states, a_next)
            target = rewards + config.discount * mask * torch.min(q_1, q_2)
            target = target.detach()

            # Compute critic loss
            q_1, q_2 = self.network.q(states, actions)
            critic_loss = F.mse_loss(q_1, target) + F.mse_loss(q_2, target)

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            if self.total_steps % config.td3_delay:
                # Compute actor loss
                action = self.network(states)
                policy_loss = -self.network.q(states, action)[0].mean()

                self.network.zero_grad()
                policy_loss.backward()
                self.network.actor_opt.step()

                # Update target network
                self.soft_update(self.target_network, self.network)