from ..network import *
from ..component import *
from ..utils import *
from .BaseAgent import *
from .DQN_agent import *


class QuantileRegressionDQNActor(DQNActor):
    def __init__(self, config):
        super().__init__(config)

    def compute_q_values(self, prediction):
        # Compute the quantile values of the prediction and take the mean over the last dimension
        # Returns a numpy array of shape (batch_size, num_actions)
        q_values = prediction['quantile'].mean(-1)
        return to_np(q_values)


class QuantileRegressionDQNAgent(DQNAgent):
    def __init__(self, config):
        self.config = config

        # Set up the replay buffer and actor
        self.replay = config.replay_fn()
        self.actor = QuantileRegressionDQNActor(config)

        # Set up the networks and optimizer
        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        # Set the network and quantile-related parameters
        self.actor.set_network(self.network)
        self.total_steps = 0
        self.batch_indices = range_tensor(config.batch_size)

        self.num_quantiles = self.config.num_quantiles
        self.quantile_weight = 1.0 / self.num_quantiles
        self.cumulative_density = tensor(
            (2 * np.arange(self.num_quantiles) + 1) / (2.0 * self.num_quantiles)).view(1, -1)

    def eval_step(self, state):
        # Compute the Q values for the given state and return the action with the highest Q value
        with self.config.state_normalizer.lock_read():
            state = self.config.state_normalizer(state)
            q_values = self.actor.compute_q_values(self.network(state))
            action = np.argmax(q_values.flatten())
        return [action]

    def compute_loss(self, transitions):
        # Compute the loss for the given transitions
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)

        # Compute the target quantile values corresponding to the next states and actions
        quantiles_next = self.target_network(next_states)['quantile'].detach()
        a_next = torch.argmax(quantiles_next.sum(-1), dim=-1)
        quantiles_next = quantiles_next[self.batch_indices, a_next, :]

        # Compute the target quantile values for the current state and action
        rewards = tensor(transitions.reward).unsqueeze(-1)
        masks = tensor(transitions.mask).unsqueeze(-1)
        quantiles_next = rewards + self.config.discount ** self.config.n_step * masks * quantiles_next
        quantiles = self.network(states)['quantile']
        actions = tensor(transitions.action).long()
        quantiles = quantiles[self.batch_indices, actions, :]

        # Compute the loss based on the quantile values and target values
        quantiles_next = quantiles_next.t().unsqueeze(-1)
        diff = quantiles_next - quantiles
        loss = huber(diff) * (self.cumulative_density - (diff.detach() < 0).float()).abs()
        return loss.sum(-1).mean(1)

    def reduce_loss(self, loss):
        # Average the loss over the batch
        return loss.mean()