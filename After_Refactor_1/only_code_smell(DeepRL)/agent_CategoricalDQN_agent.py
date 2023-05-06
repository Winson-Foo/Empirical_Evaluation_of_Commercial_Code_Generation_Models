# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)
# Permission given to modify the code as long as you keep this
# declaration at the top

from ..network import *
from ..component import tensor, range_tensor
from .BaseAgent import BaseAgent
from .DQN_agent import DQNActor, DQNAgent


class CategoricalDQNActor(DQNActor):
    def __init__(self, config):
        super().__init__(config)

    def compute_q(self, prediction):
        q_values = super().compute_q(prediction)
        return q_values


class CategoricalDQNAgent(DQNAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config.lock = mp.Lock()

        self.atoms = tensor(np.linspace(config.categorical_v_min, config.categorical_v_max, config.categorical_n_atoms))
        self.delta_atom = (config.categorical_v_max - config.categorical_v_min) / float(config.categorical_n_atoms - 1)

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
            prob_next = prob_next[range_tensor(self.config.batch_size), a_next, :]

        rewards = tensor(transitions.reward).unsqueeze(-1)
        masks = tensor(transitions.mask).unsqueeze(-1)
        atoms_target = rewards + self.config.discount ** self.config.n_step * masks * self.atoms.view(1, -1)
        atoms_target.clamp_(self.config.categorical_v_min, self.config.categorical_v_max)
        atoms_target = atoms_target.unsqueeze(1)
        target_prob = (1 - (atoms_target - self.atoms.view(1, -1, 1)).abs() / self.delta_atom).clamp(0, 1) * \
                      prob_next.unsqueeze(1)
        target_prob = target_prob.sum(-1)

        log_prob = self.network(states)['log_prob']
        actions = tensor(transitions.action).long()
        log_prob = log_prob[range_tensor(self.config.batch_size), actions, :]
        KL = (target_prob * target_prob.add(1e-5).log() - target_prob * log_prob).sum(-1)
        return KL

    def reduce_loss(self, loss):
        return loss.mean()