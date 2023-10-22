from typing import List
from torch.multiprocessing import Lock
from torch.optim import Optimizer

from ..network import Network
from ..component import StateNormalizer
from ..utils import huber, range_tensor, to_np
from .BaseAgent import BaseAgent
from .DQN_agent import DQNActor, DQNAgent


class QuantileRegressionDQNActor(DQNActor):
    """
    Actor implementation for the Quantile Regression DQN algorithm
    """

    def compute_q(self, prediction):
        """
        Compute Q values from the network's quantile outputs
        """
        q_values = prediction['quantile'].mean(-1)
        return to_np(q_values)


class QuantileRegressionDQNAgent(DQNAgent):
    """
    Implementation of the Quantile Regression DQN algorithm
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.lock = Lock()

        self.replay = config.replay_fn()
        self.actor = QuantileRegressionDQNActor(config)

        self.network: Network = config.network_fn().share_memory()
        self.target_network: Network = config.network_fn().load_state_dict(self.network.state_dict(), strict=False)
        self.optimizer: Optimizer = config.optimizer_fn(self.network.parameters())

        # Set the network for the actor
        self.actor.set_network(self.network)

        # Define class attributes for easy access to config variables
        self.num_quantiles = config.num_quantiles
        self.batch_size = config.batch_size
        self.discount = config.discount
        self.n_step = config.n_step

        # Precompute needed tensors for loss computation
        self.quantile_weight = 1.0 / self.num_quantiles
        self.cumulative_density = (2 * to_np(range_tensor(self.num_quantiles)) + 1) / (2.0 * self.num_quantiles)
        self.cumulative_density = tensor(self.cumulative_density).view(1, -1)

    def eval_step(self, state) -> List[int]:
        """
        Compute the action to take given the current state
        """
        with self.config.state_normalizer.set_read_only():
            # Normalize the state
            state = self.config.state_normalizer(state)

            # Compute Q values and return the action with highest Q value
            q = self.network(state)['quantile'].mean(-1)
            action = int(to_np(q.argmax()))
        return [action]

    def compute_loss(self, transitions):
        """
        Compute the quantile regression loss for a batch of transitions
        """
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)
        rewards = tensor(transitions.reward).unsqueeze(-1)
        masks = tensor(transitions.mask).unsqueeze(-1)
        actions = tensor(transitions.action).long()

        # Compute the quantile regression targets
        with torch.no_grad():
            quantiles_next = self.target_network(next_states)['quantile']
            a_next = torch.argmax(quantiles_next.sum(-1), dim=-1)
            quantiles_next = quantiles_next[range(len(quantiles_next)), a_next, :]
            quantiles_next = rewards + self.discount ** self.n_step * masks * quantiles_next

        # Compute the quantile regression predictions and the quantile values for the actions taken
        quantiles = self.network(states)['quantile']
        quantiles = quantiles[range(len(quantiles)), actions, :]

        # Compute the loss using elementwise Huber loss and the target and prediction cumulative distributions
        quantiles_next = quantiles_next.t().unsqueeze(-1)
        diff = quantiles_next - quantiles
        loss = huber(diff) * (self.cumulative_density - (diff.detach() < 0).float()).abs()

        # Return the mean loss over the batch
        return loss.sum(-1).mean(1)

    def reduce_loss(self, loss):
        """
        Reduce the loss to a scalar value
        """
        return loss.mean()