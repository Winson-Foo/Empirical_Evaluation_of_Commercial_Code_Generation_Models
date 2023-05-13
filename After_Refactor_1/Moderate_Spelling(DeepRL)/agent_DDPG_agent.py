from typing import Dict, Any, Tuple

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from ..network import ActorNetwork, CriticNetwork
from ..replay import Replay
from .BaseAgent import BaseAgent


class DDPGAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.task = config.task_fn()
        self.state_normalizer = config.state_normalizer_fn()

        # Create actor and critic networks
        self.actor_network = ActorNetwork(self.task.state_shape, self.task.action_shape)
        self.critic_network = CriticNetwork(self.task.state_shape, self.task.action_shape)

        # Create target actor and critic networks
        self.target_actor_network = ActorNetwork(self.task.state_shape, self.task.action_shape)
        self.target_critic_network = CriticNetwork(self.task.state_shape, self.task.action_shape)

        # Copy parameters from actor and critic networks to target actor and critic networks
        self.target_actor_network.load_state_dict(self.actor_network.state_dict())
        self.target_critic_network.load_state_dict(self.critic_network.state_dict())

        self.replay = Replay(config.replay_size)

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=config.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=config.critic_learning_rate)

        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None

    def update_target_networks(self, target: torch.nn.Module, src: torch.nn.Module, mix_factor: float):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - mix_factor) + param * mix_factor)

    def evaluate(self, state: np.ndarray) -> np.ndarray:
        state = self.state_normalizer(state)
        action = self.actor_network(state)
        return action.detach().numpy()

    def train(self) -> None:
        # Sample a random minibatch of transitions from replay buffer
        transitions = self.replay.sample(self.config.batch_size)
        states = torch.tensor(transitions.state, dtype=torch.float32)
        actions = torch.tensor(transitions.action, dtype=torch.float32)
        rewards = torch.tensor(transitions.reward, dtype=torch.float32)
        next_states = torch.tensor(transitions.next_state, dtype=torch.float32)
        masks = torch.tensor(transitions.mask, dtype=torch.float32)

        # Compute target Q values
        with torch.no_grad():
            next_actions = self.target_actor_network(next_states)
            next_Q = self.target_critic_network(next_states, next_actions)
            target_Q = rewards + (self.config.gamma * masks * next_Q)

        # Compute critic loss
        Q = self.critic_network(states, actions)
        critic_loss = F.mse_loss(Q, target_Q)

        # Update critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        policy_loss = -self.critic_network(states, self.actor_network(states)).mean()

        # Update actor network
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.update_target_networks(self.target_actor_network, self.actor_network, self.config.tau)
        self.update_target_networks(self.target_critic_network, self.critic_network, self.config.tau)

    def step(self) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = self.state_normalizer(self.state)

        if self.total_steps < self.config.warmup_steps:
            # Random action during the warm-up period
            action = self.task.action_space.sample()
        else:
            # Select an action using the actor network, adding exploration noise using the random process
            action = self.actor_network(self.state).detach().numpy()
            action += self.random_process.sample()

        # Clip the action within the action space bounds
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)

        # Take a step in the environment
        next_state, reward, done, info = self.task.step(action)
        next_state = self.state_normalizer(next_state)
        reward = self.config.reward_normalizer(reward)

        # Add transition to replay buffer
        self.replay.add_transition(self.state, action, reward, next_state, done)

        # Reset random process if episode has terminated
        if done[0]:
            self.random_process.reset_states()

        # Update state and total step count
        self.state = next_state
        self.total_steps += 1

        if self.total_steps > self.config.warmup_steps and self.total_steps % self.config.policy_update_freq == 0:
            for _ in range(self.config.policy_update_repeat):
                self.train()

        return action, reward, done, info