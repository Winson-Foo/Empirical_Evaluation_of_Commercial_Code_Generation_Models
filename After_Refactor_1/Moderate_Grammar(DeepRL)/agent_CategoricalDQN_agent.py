from typing import List
import torch.multiprocessing as mp
from torch import Tensor
from numpy import linspace
from ..network import Network
from ..component import Normalizer, Transition, ReplayBuffer
from ..utils import range_tensor, to_np

from .BaseAgent import BaseAgent
from .DQN_agent import DQNAgent, DQNActor


class CategoricalDQNActor(DQNActor):
    def __init__(self, params):
        super().__init__(params)

    def _set_up(self):
        self.params.atoms = Tensor(self.params.atoms)

    def compute_q(self, prediction: dict) -> List[float]:
        q_values = (prediction['prob'] * self.params.atoms).sum(-1)
        return to_np(q_values)


class CategoricalDQNAgent(DQNAgent):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.replay = self.params.replay_fn()
        self.actor = CategoricalDQNActor(self.params)
        self.network = self.params.network_fn()
        self.network.share_memory()
        self.target_network = self.params.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = self.params.optimizer_fn(self.network.parameters())
        self.actor.set_network(self.network)
        self.total_steps = 0
        self.batch_indices = range_tensor(self.params.batch_size)
        self.atoms = Tensor(linspace(self.params.categorical_v_min, self.params.categorical_v_max, self.params.categorical_n_atoms))
        self.delta_atom = (self.params.categorical_v_max - self.params.categorical_v_min) / float(self.params.categorical_n_atoms - 1)

    def eval_step(self, state: Tensor) -> int:
        self.params.state_normalizer.set_read_only()
        state = self.params.state_normalizer(state)
        prediction = self.network(state)
        q = (prediction['prob'] * self.atoms).sum(-1)
        action = to_np(q.argmax(-1))
        self.params.state_normalizer.unset_read_only()
        return action

    def compute_loss(self, transitions: Transition) -> Tensor:
        states = self.params.state_normalizer(transitions.state)
        next_states = self.params.state_normalizer(transitions.next_state)

        with torch.no_grad():
            prob_next = self.target_network(next_states)['prob']
            q_next = (prob_next * self.atoms).sum(-1)
            if self.params.double_q:
                a_next = torch.argmax((self.network(next_states)['prob'] * self.atoms).sum(-1), dim=-1)
            else:
                a_next = torch.argmax(q_next, dim=-1)

            prob_next = prob_next[self.batch_indices, a_next, :]

        rewards = Tensor(transitions.reward).unsqueeze(-1)
        masks = Tensor(transitions.mask).unsqueeze(-1)
        atoms_target = rewards + self.params.discount ** self.params.n_step * masks * self.atoms.view(1, -1)

        atoms_target.clamp_(self.params.categorical_v_min, self.params.categorical_v_max)
        atoms_target = atoms_target.unsqueeze(1)

        target_prob = (1 - (atoms_target - self.atoms.view(1, -1, 1)).abs() / self.delta_atom).clamp(0, 1) * \
                      prob_next.unsqueeze(1)

        target_prob = target_prob.sum(-1)
        log_prob = self.network(states)['log_prob']
        actions = Tensor(transitions.action).long()
        log_prob = log_prob[self.batch_indices, actions, :]

        KL = (target_prob * target_prob.add(1e-5).log() - target_prob * log_prob).sum(-1)
        return KL

    def reduce_loss(self, loss: Tensor) -> Tensor:
        return loss.mean()