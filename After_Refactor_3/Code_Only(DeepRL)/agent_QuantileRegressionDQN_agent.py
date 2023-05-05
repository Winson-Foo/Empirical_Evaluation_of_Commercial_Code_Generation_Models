# Refactored code:

from ..network import *
from ..component import *
from ..utils import *
from .BaseAgent import *
from .DQN_agent import *

class QuantileRegressionDQNActor(DQNActor):
    """
    Class to represent an actor in an agent making use of a Quantile Regression DQN.
    """
    def __init__(self, config):
        super().__init__(config)

    def compute_q(self, prediction):
        """Compute Q-values from network predictions"""
        q_values = prediction['quantile'].mean(-1)
        return to_np(q_values)


class QuantileRegressionDQNAgent(DQNAgent):
    """
    Class to represent an agent making use of a Quantile Regression DQN.
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = QuantileRegressionDQNActor(config)

        # Initialize network and optimizer
        self.network = config.network_fn()
        self.network.share_memory()

        # Make a target network with the same weight as the main network
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.actor.set_network(self.network)

        # Initialize other variables
        self.total_steps = 0
        self.batch_indices = range_tensor(config.batch_size)

        self.quantile_weight = 1.0 / self.config.num_quantiles
        self.cumulative_density = tensor(
            (2 * np.arange(self.config.num_quantiles) + 1) / (2.0 * self.config.num_quantiles)).view(1, -1)

    def eval_step(self, state):
        """Evaluation step for the agent"""
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)['quantile'].mean(-1)
        action = np.argmax(to_np(q).flatten())
        self.config.state_normalizer.unset_read_only()
        return [action]

    def compute_loss(self, transitions):
        """Compute the TD-error loss of the agent"""
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)

        # Compute quantiles of the next state
        quantiles_next = self.target_network(next_states)['quantile'].detach()
        a_next = torch.argmax(quantiles_next.sum(-1), dim=-1)
        quantiles_next = quantiles_next[self.batch_indices, a_next, :]

        # Compute the estimated Q-values using the quantiles and Bellman equation
        rewards = tensor(transitions.reward).unsqueeze(-1)
        masks = tensor(transitions.mask).unsqueeze(-1)
        quantiles_next = rewards + self.config.discount ** self.config.n_step * masks * quantiles_next

        # Compute the predicted quantiles for the given state and actions
        quantiles = self.network(states)['quantile']
        actions = tensor(transitions.action).long()
        quantiles = quantiles[self.batch_indices, actions, :]

        # Compute the TD-error loss 
        quantiles_next = quantiles_next.t().unsqueeze(-1)
        diff = quantiles_next - quantiles
        loss = huber(diff) * (self.cumulative_density - (diff.detach() < 0).float()).abs()
        return loss.sum(-1).mean(1)

    def reduce_loss(self, loss):
        """Compute mean loss across all processes"""
        return loss.mean()