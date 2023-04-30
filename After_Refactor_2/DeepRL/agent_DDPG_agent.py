from typing import Optional

import numpy as np
import torch
from torch import nn, optim

from components import ReplayBuffer, RandomProcess, Config


class DDPGAgent:
    """
    An agent that uses the Deep Deterministic Policy Gradient (DDPG) algorithm to learn how to solve
    a task.

    :param config: A configuration object containing parameters for the agent and components.
    """

    def __init__(self, config: Config):
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay_buffer = config.replay_buffer_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None

    def soft_update(self, target: nn.Module, src: nn.Module) -> None:
        """
        Performs a soft update of the parameters of a target network using the parameters of a source network.

        :param target: The target network to update.
        :param src: The source network to copy the parameters from.
        """

        for target_param, src_param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1 - self.config.target_network_mix) + src_param * self.config.target_network_mix)

    def eval_step(self, state: np.ndarray) -> np.ndarray:
        """
        Performs a single evaluation step with the agent.

        :param state: The current state of the environment.
        :return: The agent's action in response to the given state.
        """

        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()

        return torch.Tensor.cpu(action).detach().numpy()

    def step(self) -> Optional[float]:
        """
        Performs a single learning step with the agent.

        :return: The total reward obtained during this learning step, or None if the episode hasn't ended.
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
            action = torch.Tensor.cpu(action).detach().numpy()
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
            mask=1 - np.asarray(done, dtype=np.int32),
        ))

        if done[0]:
            self.random_process.reset_states()
            self.state = None
            return self.get_total_reward()

        self.state = next_state
        self.total_steps += 1

        if self.replay_buffer.size() >= config.warm_up:
            transitions = self.replay_buffer.sample()
            states = torch.Tensor(transitions.state)
            actions = torch.Tensor(transitions.action)
            rewards = torch.Tensor(transitions.reward).unsqueeze(-1)
            next_states = torch.Tensor(transitions.next_state)
            mask = torch.Tensor(transitions.mask).unsqueeze(-1)

            phi_next = self.target_network.feature(next_states)
            a_next = self.target_network.actor(phi_next)
            q_next = self.target_network.critic(phi_next, a_next)
            q_next = config.discount * mask * q_next
            q_next.add_(rewards)
            q_next = q_next.detach()

            phi = self.network.feature(states)
            q = self.network.critic(phi, actions)
            critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            phi = self.network.feature(states)
            action = self.network.actor(phi)
            policy_loss = -self.network.critic(phi.detach(), action).mean()

            self.network.zero_grad()
            policy_loss.backward()
            self.network.actor_opt.step()

            self.soft_update(self.target_network, self.network)

        return None

    def record_online_return(self, info: dict) -> None:
        """
        Records the online return specified in `info` to the logger.

        :param info: A dictionary containing information about the episode.
        """

        if "episode" in info:
            self.config.logger.add_scalar("online_return", info["episode"]["return"], self.total_steps)

    def get_total_reward(self) -> float:
        """
        Returns the total reward obtained during the current episode.

        :return: The total reward obtained during the current episode.
        """

        return self.task.get_episode_reward() if hasattr(self.task, "get_episode_reward") else 0.0