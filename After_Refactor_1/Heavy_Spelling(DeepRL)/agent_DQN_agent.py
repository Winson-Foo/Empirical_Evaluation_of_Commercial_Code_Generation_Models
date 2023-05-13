import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp

from collections import namedtuple
from typing import List, Tuple

from ..component import *
from ..utils import *
from .BaseAgent import *
from ..network import *


# Named tuple to represent a transition in the replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'info'))


class DQNActor(BaseActor):
    """
    Actor that interacts with the environment and collects transitions
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.start()

    def _transition(self) -> Transition:
        """
        Sample an action from the policy and record the resulting transition
        """
        if self.state is None:
            self.state = self.task.reset()
        config = self.config
        if config.noisy_linear:
            self.network.reset_noise()
        with config.lock:
            prediction = self.network(config.state_normalizer(self.state))
        q_values = self._compute_q_values(prediction)

        if config.noisy_linear:
            epsilon = 0
        elif self.total_steps < config.exploration_steps:
            epsilon = 1
        else:
            epsilon = config.random_action_prob()
        action = epsilon_greedy(epsilon, q_values)
        next_state, reward, done, info = self.task.step(action)
        entry = Transition(self.state, action, reward, next_state, done, info)
        self.total_steps += 1
        self.state = next_state
        return entry

    def _compute_q_values(self, prediction) -> List[float]:
        """
        Extract the q_values from the prediction output
        """
        return to_np(prediction['q'])


class DQNAgent(BaseAgent):
    """
    DQN algorithm implementation
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        config.lock = mp.Lock()

        # Initialize actor and replay buffer
        self.actor = DQNActor(config)
        self.replay_buffer = config.replay_fn()

        # Initialize network and optimizer
        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        # Set the actor network
        self.actor.set_network(self.network)

        self.total_steps = 0

    def close(self):
        close_obj(self.replay_buffer)
        close_obj(self.actor)

    def eval_step(self, state):
        """
        Perform a greedy action selection based on the current policy
        """
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)['q']
        action = to_np(q.argmax(-1))
        self.config.state_normalizer.unset_read_only()
        return action

    def compute_loss(self, transitions) -> torch.Tensor:
        """
        Compute the MSE loss between the predicted q_values and the target q_values
        """
        config = self.config
        states = config.state_normalizer(transitions.state)
        next_states = config.state_normalizer(transitions.next_state)
        with torch.no_grad():
            q_next = self.target_network(next_states)['q'].detach()
            if config.double_q:
                best_actions = torch.argmax(self.network(next_states)['q'], dim=-1)
                q_next = q_next.gather(1, best_actions.unsqueeze(-1)).squeeze(1)
            else:
                q_next = q_next.max(1)[0]
        masks = torch.tensor(transitions.done, dtype=torch.float32)
        rewards = torch.tensor(transitions.reward, dtype=torch.float32)
        q_targets = rewards + config.discount ** config.n_step * q_next * (1 - masks)
        actions = torch.tensor(transitions.action, dtype=torch.long)
        q_values = self.network(states)['q']
        q = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = q_targets - q
        return loss.pow(2).mul(0.5).mean()

    def train_step(self):
        """
        Train the network on a batch of transitions from the replay buffer
        """
        config = self.config
        transitions = self.replay_buffer.sample()
        if config.noisy_linear:
            self.target_network.reset_noise()
            self.network.reset_noise()
        loss = self.compute_loss(transitions)
        if isinstance(transitions, PrioritizedTransition):
            priorities = loss.abs().add(config.replay_eps).pow(config.replay_alpha)
            idxs = torch.tensor(transitions.idx, dtype=torch.long)
            self.replay_buffer.update_priorities(zip(to_np(idxs), to_np(priorities)))
            sampling_probs = torch.tensor(transitions.sampling_prob, dtype=torch.float32)
            weights = sampling_probs.mul(sampling_probs.size(0)).add(1e-6).pow(-config.replay_beta())
            weights = weights / weights.max()
            loss = loss.mul(weights)

        loss = self.reduce_loss(loss)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        with config.lock:
            self.optimizer.step()

        if self.total_steps / config.sgd_update_frequency % \
                config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        return loss

    def train(self):
        """
        Train the network for the specified number of steps
        """
        for i in range(self.config.max_steps):
            self.step()
            if (i + 1) % self.config.log_interval == 0:
                self.log_metrics()

    def step(self):
        """
        Collect a transition from the actor and add it to the replay buffer
        """
        transitions = self.actor.step()
        for transition in transitions:
            self.record_online_return(transition.info)
            self.total_steps += 1
            state = LazyFrames.flatten(transition.state) if isinstance(transition.state, LazyFrames) else transition.state
            self.replay_buffer.add(state, transition.action, transition.reward, transition.next_state, transition.done)

        if self.total_steps > self.config.exploration_steps:
            self.train_step()

        if self.total_steps > 0 and self.total_steps % self.config.log_interval == 0:
            self.log_metrics()

    def reduce_loss(self, loss) -> torch.Tensor:
        """
        Compute the reduced mean of the loss tensor
        """
        return loss.mean()