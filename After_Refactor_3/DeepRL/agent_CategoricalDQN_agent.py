# Imports
import torch
import torch.multiprocessing as mp
import numpy as np

from ..network import *
from ..component import *
from ..utils import *
from .BaseAgent import *
from .DQN_agent import *


# Categorical DQN actor class
class CategoricalDQNActor(DQNActor):
    def __init__(self, config):
        super().__init__(config)
        self.config.atoms = tensor(self.config.atoms)
        
    def compute_q(self, prediction):
        q_values = (prediction['prob'] * self.config.atoms).sum(-1)
        return to_np(q_values)


# Categorical DQN agent class
class CategoricalDQNAgent(DQNAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config.lock = mp.Lock()
        self.config.atoms = np.linspace(self.config.categorical_v_min,
                                        self.config.categorical_v_max, self.config.categorical_n_atoms)
        
        self.replay = self.config.replay_fn()
        self.actor = CategoricalDQNActor(self.config)
        
        self.network = self.config.network_fn()
        self.network.share_memory()
        self.target_network = self.config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = self.config.optimizer_fn(self.network.parameters())
        
        self.actor.set_network(self.network)
        
        self.total_steps = 0
        self.batch_indices = range_tensor(self.config.batch_size)
        self.atoms = tensor(self.config.atoms)
        self.delta_atom = (self.config.categorical_v_max - self.config.categorical_v_min) / \
                            float(self.config.categorical_n_atoms - 1)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        prediction = self.network(state)
        q = (prediction['prob'] * self.atoms).sum(-1)
        action = to_np(q.argmax(-1))
        self.config.state_normalizer.unset_read_only()
        return action

    def compute_loss(self, transitions):
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)
        with torch.no_grad():
            prob_next = self.target_network(next_states)['prob']
            q_next = (prob_next * self.atoms).sum(-1)
            if self.config.double_q:
                a_next = torch.argmax((self.network(next_states)['prob'] * self.atoms).sum(-1), dim=-1)
            else:
                a_next = torch.argmax(q_next, dim=-1)
            prob_next = prob_next[self.batch_indices, a_next, :]

        rewards = tensor(transitions.reward).unsqueeze(-1)
        masks = tensor(transitions.mask).unsqueeze(-1)
        atoms_target = rewards + self.config.discount ** self.config.n_step * masks * \
                        self.atoms.view(1, -1)
        atoms_target.clamp_(self.config.categorical_v_min, self.config.categorical_v_max)
        atoms_target = atoms_target.unsqueeze(1)
        target_prob = (1 - (atoms_target - self.atoms.view(1, -1, 1)).abs() / self.delta_atom).clamp(0, 1) * \
                        prob_next.unsqueeze(1)
        target_prob = target_prob.sum(-1)

        log_prob = self.network(states)['log_prob']
        actions = tensor(transitions.action).long()
        log_prob = log_prob[self.batch_indices, actions, :]
        KL = (target_prob * target_prob.add(1e-5).log() - target_prob * log_prob).sum(-1)
        return KL

    def reduce_loss(self, loss):
        return loss.mean()
