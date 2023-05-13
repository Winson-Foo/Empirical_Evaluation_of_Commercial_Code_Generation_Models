# Improved and Refactored Code with Docstrings

"""
DQN Agent for Reinforcement Learning
"""

import time
from multiprocessing import Lock
from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ..network import BaseNetwork
from ..component import BaseActor, BaseAgent
from ..replay import PrioritizedTransition, BaseReplayBuffer
from ..utils import to_np, epsilon_greedy, tensor, close_obj
from ..wrappers import LazyFrames


class DQNActor(BaseActor):
    """
    Actor for DQNAgent

    Args:
    config (argparse.Namespace): Configuration object, obtained from argument parser
    """
    
    def __init__(self, config: object) -> None:
        super().__init__(config)
        self.config = config
        self.start()
        
    def compute_q(self, prediction: dict) -> np.ndarray:
        """Computes the state-action values (Q-values) using the prediction dictionary"""
        
        q_values = to_np(prediction['q'])
        return q_values
    
    def _transition(self) -> Tuple[Union[LazyFrames, np.ndarray], int, float, Union[LazyFrames, np.ndarray], bool, dict]:
        """
        Performs a single transition
        Returns tuple (state, action, reward, next_state, done, info)
        """
        
        # If state is none, reset it
        if self._state is None:
            self._state = self._task.reset()
        
        # Reset network noise, if used
        if self.config.noisy_linear:
            self._network.reset_noise()
        
        # Predict Q-values using the state and the network
        with self.config.lock:
            prediction = self._network(self.config.state_normalizer(self._state))
        q_values = self.compute_q(prediction)
        
        # Calculate epsilon greedy action
        if config.noisy_linear:
            epsilon = 0
        elif self._total_steps < self.config.exploration_steps:
            epsilon = 1
        else:
            epsilon = self.config.random_action_prob()
        action = epsilon_greedy(epsilon, q_values)
        next_state, reward, done, info = self._task.step(action)
        
        # Return the transition as tuple
        entry = [self._state, action, reward, next_state, done, info]
        self._total_steps += 1
        self._state = next_state
        return entry


class DQNAgent(BaseAgent):
    """
    DQN Agent Implementation

    Args:
    config (argparse.Namespace): Configuration object, obtained from argument parser
    """
    
    def __init__(self, config: object) -> None:
        super().__init__(config)
        self.config = config
        config.lock = Lock()

        self.replay = config.replay_fn()
        self.actor = DQNActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)
        self.total_steps = 0

    def close(self) -> None:
        """Closes the replay buffer and actor objects"""
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state: Union[LazyFrames, np.ndarray]) -> int:
        """
        Evaluates the Q-values for the given state and returns the greedy action
        
        Args:
        state (LazyFrames or ndarray): The state for which Q-values need to be evaluated
        
        Returns:
        int: Greedy action for the given state
        """
        
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)['q']
        action = to_np(q.argmax(-1))
        self.config.state_normalizer.unset_read_only()
        return action

    def reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Reduces the loss tensor"""
        return loss.pow(2).mul(0.5).mean()

    def compute_loss(self, transitions: Union[BaseReplayBuffer, PrioritizedTransition]) -> torch.Tensor:
        """
        Computes the loss tensor for a batch of transitions
        
        Args:
        transitions (BaseReplayBuffer or PrioritizedTransition): Batch of transitions sampled from the replay buffer
        
        Returns:
        torch.Tensor: Loss tensor computed for the given transitions
        """
        
        config = self.config
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)
        with torch.no_grad():
            q_next = self.target_network(next_states)['q'].detach()
            if self.config.double_q:
                best_actions = torch.argmax(self.network(next_states)['q'], dim=-1)
                q_next = q_next.gather(1, best_actions.unsqueeze(-1)).squeeze(1)
            else:
                q_next = q_next.max(1)[0]
        masks = tensor(transitions.mask)
        rewards = tensor(transitions.reward)
        q_target = rewards + self.config.discount ** config.n_step * q_next * masks
        actions = tensor(transitions.action).long()
        q = self.network(states)['q']
        q = q.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = q_target - q
        return loss

    def step(self) -> None:
        """Performs a single update step"""
        
        config = self.config
        transitions = self.actor.step()
        for states, actions, rewards, next_states, dones, info in transitions:
            self.record_online_return(info)
            self.total_steps += 1
            self.replay.feed(dict(
                state=np.array([s[-1] if isinstance(s, LazyFrames) else s for s in states]),
                action=actions,
                reward=[config.reward_normalizer(r) for r in rewards],
                mask=1 - np.asarray(dones, dtype=np.int32),
            ))

        if self.total_steps > self.config.exploration_steps:
            transitions = self.replay.sample()
            
            # Reset network noise, if used
            if config.noisy_linear:
                self.target_network.reset_noise()
                self.network.reset_noise()
                
            # Compute loss and update network weights
            loss = self.compute_loss(transitions)
            if isinstance(transitions, PrioritizedTransition):
                priorities = loss.abs().add(config.replay_eps).pow(config.replay_alpha)
                idxs = tensor(transitions.idx).long()
                self.replay.update_priorities(zip(to_np(idxs), to_np(priorities)))
                sampling_probs = tensor(transitions.sampling_prob)
                weights = sampling_probs.mul(sampling_probs.size(0)).add(1e-6).pow(-config.replay_beta())
                weights = weights / weights.max()
                loss = loss.mul(weights)

            loss = self.reduce_loss(loss)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            with config.lock:
                self.optimizer.step()

        if self.total_steps / self.config.sgd_update_frequency % self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())