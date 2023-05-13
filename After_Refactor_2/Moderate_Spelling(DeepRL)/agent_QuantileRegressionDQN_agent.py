from typing import List, Tuple
import torch.multiprocessing as mp

from ..network import Network
from ..component import StateNormalizer
from ..utils import to_np, range_tensor, tensor, huber
from .BaseAgent import BaseAgent
from .DQN_agent import DQNActor, DQNAgent


class QuantileRegressionDQNActor(DQNActor):
    def __init__(self, config):
        super().__init__(config)

    def compute_q(self, prediction):
        q_values = prediction['quantile'].mean(-1)
        return to_np(q_values)


class QuantileRegressionDQNAgent(DQNAgent):
    def __init__(self, config):
        super().__init__(config)

        self.replay = config.replay_fn()
        self.actor = QuantileRegressionDQNActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)

        self.total_steps = 0
        self.batch_indices = range_tensor(config.batch_size)

        self.quantile_weight = 1.0 / self.config.num_quantiles
        self.cumulative_density = tensor(
            (2 * np.arange(self.config.num_quantiles) + 1) / (2.0 * self.config.num_quantiles)).view(1, -1)

    def eval_step(self, state: List[float]) -> List[int]:
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)['quantile'].mean(-1)
        action = int(torch.argmax(to_np(q).flatten()))
        self.config.state_normalizer.unset_read_only()
        return [action]

    def compute_loss(self, transitions) -> torch.Tensor:
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)

        quantiles_next = self.target_network(next_states)['quantile'].detach()
        a_next = torch.argmax(quantiles_next.sum(-1), dim=-1)
        quantiles_next = quantiles_next[self.batch_indices, a_next, :]

        rewards = tensor(transitions.reward).unsqueeze(-1)
        masks = tensor(transitions.mask).unsqueeze(-1)
        quantiles_next = rewards + self.config.discount ** self.config.n_step * masks * quantiles_next

        quantiles = self.network(states)['quantile']
        actions = tensor(transitions.action).long()
        quantiles = quantiles[self.batch_indices, actions, :]

        quantiles_next = quantiles_next.t().unsqueeze(-1)
        diff = quantiles_next - quantiles
        loss = huber(diff) * (self.cumulative_density - (diff.detach() < 0).float()).abs()
        return loss.sum(-1).mean(1)

    def reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss.mean()


class QuantileRegressionDQNConfig:
    def __init__(self,
                 network_fn,
                 optimizer_fn,
                 state_normalizer: StateNormalizer,
                 replay_fn,
                 discount,
                 n_step,
                 num_quantiles,
                 batch_size):
        self.network_fn = network_fn
        self.optimizer_fn = optimizer_fn
        self.state_normalizer = state_normalizer
        self.replay_fn = replay_fn
        self.discount = discount
        self.n_step = n_step
        self.num_quantiles = num_quantiles
        self.batch_size = batch_size


def compute_quantiles(x: torch.Tensor, num_quantiles: int) -> Tuple[torch.Tensor, torch.Tensor]:
    tau = torch.arange(0, num_quantiles, 1, dtype=torch.float32).unsqueeze(-1).to(x.device) / num_quantiles + 0.5 / num_quantiles
    x = x.unsqueeze(1)
    tau = tau.unsqueeze(0)
    u = tau - (x < 0).type(torch.float32)
    return u.clamp(min=0), u.clamp(max=1) - u.clamp(min=0)


if __name__ == '__main__':
    def network_fn():
        return Network(input_dim, output_dim)

    def optimizer_fn(params):
        return torch.optim.Adam(params, lr=lr)

    def replay_fn():
        return Replay(memory_size)

    state_normalizer = StateNormalizer(input_dim)

    config = QuantileRegressionDQNConfig(network_fn, optimizer_fn, state_normalizer, replay_fn, discount, n_step, num_quantiles, batch_size)

    agent = QuantileRegressionDQNAgent(config)

    # train the agent...