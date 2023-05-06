from typing import Tuple
import torch.nn.functional as F

from torch import Tensor
from torch.optim import Optimizer

from ..network import Network, QuantileNetwork
from ..component import ReplayBuffer
from ..utils import to_np, range_tensor, tensor

from .base_agent import BaseAgent


class QuantileRegressionDQNActor:
    """
    Class for Quantile Regression DQN actor
    """

    def __init__(self, network: Network):
        self.network = network

    def get_action(self, state: Tensor) -> int:
        """
        Returns the action with highest Q-value from current state
        """
        q = self.network(state)['quantile'].mean(-1)
        action = to_np(q).argmax()
        return action


class QuantileRegressionDQNAgent(BaseAgent):
    """
    Class for Quantile Regression DQN agent
    """

    def __init__(self,
                 network_fn: Callable[[], QuantileNetwork],
                 optimizer_fn: Callable[[Network], Optimizer],
                 replay_fn: Callable[[], ReplayBuffer],
                 state_normalizer,
                 discount: float = 0.99,
                 n_step: int = 3,
                 batch_size: int = 32,
                 num_quantiles: int = 8):
        super().__init__()
        self.network_fn = network_fn
        self.optimizer_fn = optimizer_fn
        self.replay_fn = replay_fn
        self.state_normalizer = state_normalizer
        self.discount = discount
        self.batch_size = batch_size
        self.n_step = n_step
        self.num_quantiles = num_quantiles
        self.batch_indices = range_tensor(batch_size)

        # Lock for updating network parameters
        self.lock = mp.Lock()

        # Create replay buffer and actor
        self.replay = self.replay_fn()
        self.network = self.network_fn()
        self.target_network = self.network_fn()
        self.actor = QuantileRegressionDQNActor(self.network)

        # Copy over network parameters to target network
        self.target_network.load_state_dict(self.network.state_dict())

        # Create optimizer for network
        self.optimizer = self.optimizer_fn(self.network.parameters())

        # Calculate Cumulative Density for Quantiles
        density = (2 * torch.arange(self.num_quantiles).float() + 1) / (2.0 * self.num_quantiles)
        self.cumulative_density = tensor(density).unsqueeze(0)

    def learn(self, state: Tensor, action: int, reward: float, next_state: Tensor, done: bool) -> float:
        # Normalize states
        state = self.state_normalizer(state)
        next_state = self.state_normalizer(next_state)

        # Add transition to replay buffer
        self.replay.add(state, action, reward, next_state, done)

        loss = 0.0

        # Check if enough samples are present in replay buffer to learn
        if len(self.replay) > self.batch_size:

            # Sample a batch from the replay buffer
            transitions = self.replay.sample(self.batch_size)

            # Calculate loss
            loss = self._calculate_loss(transitions)

            # Gradient update on network parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update target network parameters
            self._update_target_network()

        return loss.detach().item()

    def get_action(self, state: Tensor) -> int:
        self.state_normalizer.set_read_only()
        action = self.actor.get_action(state)
        self.state_normalizer.unset_read_only()
        return action

    def _calculate_loss(self, transitions: Tuple) -> Tensor:
        """
        Helper function to calculate loss
        """
        # Convert numpy arrays to tensors
        state, action, reward, next_state, done = transitions

        # Convert to tensors
        state = tensor(state)
        next_state = tensor(next_state)
        reward = tensor(reward).unsqueeze(-1)
        done = tensor(done).unsqueeze(-1)

        # Compute Quantile values for next state
        quantiles_next = self.target_network(next_state)['quantile'].detach()
        a_next = torch.argmax(quantiles_next.sum(-1), dim=-1)
        quantiles_next = quantiles_next[self.batch_indices, a_next, :]

        # Calculate expected Quantile value
        quantiles = self.network(state)['quantile']
        actions = tensor(action).long()
        quantiles = quantiles[self.batch_indices, actions, :]

        # Calculate difference in Quantile values
        quantiles_next = quantiles_next.t().unsqueeze(-1)
        diff = quantiles_next - quantiles

        # Compute modified huber loss
        tau = ((torch.arange(self.num_quantiles).float() + 0.5) / self.num_quantiles).unsqueeze(-1).unsqueeze(-1)
        error = diff.detach() < 0
        huber_loss = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction='none')
        loss = (tau - error).abs() * huber_loss
        loss = loss.mean(-1).sum(-1).mean()

        # Return loss
        return loss

    def _update_target_network(self):
        """
        Helper function to update target network parameters
        """
        with self.lock:
            self.target_network.load_state_dict(self.network.state_dict())