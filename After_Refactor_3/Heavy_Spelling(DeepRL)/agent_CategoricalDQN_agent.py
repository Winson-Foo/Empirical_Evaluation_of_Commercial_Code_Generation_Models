import torch
import torch.multiprocessing as mp
import numpy as np

from ..network import Network
from ..component import to_np, range_tensor, tensor
from ..utils import time_it
from .BaseAgent import BaseAgent


class CategoricalDQNAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.update_step = 0

        self._set_up()
        self._init_networks()
        self._init_replay()

    def _set_up(self):
        self.config.atoms = tensor(self.config.atoms)

    def _init_networks(self):
        self.network = Network(self.config.state_dim, self.config.action_dim, self.config.hidden_dim).to(self.device)
        self.target_network = Network(self.config.state_dim, self.config.action_dim, self.config.hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

    def _init_replay(self):
        self.replay = self.config.replay_fn()

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        prediction = self.network(state)
        q = (prediction['prob'] * self.config.atoms).sum(-1)
        action = to_np(q.argmax(-1))
        self.config.state_normalizer.unset_read_only()
        return action

    def compute_loss(self, transitions):
        states, next_states, rewards, masks, actions = self._prepare_data(transitions)

        with torch.no_grad():
            prob_next = self.target_network(next_states)['prob']
            q_next = (prob_next * self.config.atoms).sum(-1)

            if self.config.double_q:
                a_next = torch.argmax((self.network(next_states)['prob'] * self.config.atoms).sum(-1), dim=-1)
            else:
                a_next = torch.argmax(q_next, dim=-1)

            prob_next = prob_next[self.batch_indices, a_next, :]

        atoms_target = self._compute_atoms_target(rewards, masks)
        target_prob = self._compute_target_prob(atoms_target, prob_next)

        log_prob = self.network(states)['log_prob']
        log_prob = log_prob[self.batch_indices, actions, :]
        KL = self._compute_kl_divergence(target_prob, log_prob)

        return KL

    def _prepare_data(self, transitions):
        config = self.config

        states = self.config.state_normalizer(transitions.state).to(self.device)
        next_states = self.config.state_normalizer(transitions.next_state).to(self.device)
        rewards = tensor(transitions.reward).unsqueeze(-1).to(self.device)
        masks = tensor(transitions.mask).unsqueeze(-1).to(self.device)
        actions = tensor(transitions.action).long().to(self.device)

        self.batch_indices = range_tensor(config.batch_size).to(self.device)

        return states, next_states, rewards, masks, actions

    def _compute_atoms_target(self, rewards, masks):
        atoms_target = rewards + self.config.discount ** self.config.n_step * masks * self.config.atoms.view(1, -1)
        atoms_target = torch.clamp(atoms_target, self.config.categorical_v_min, self.config.categorical_v_max)
        atoms_target = atoms_target.unsqueeze(1)
        return atoms_target

    def _compute_target_prob(self, atoms_target, prob_next):
        delta_atom = (self.config.categorical_v_max - self.config.categorical_v_min) / float(self.config.categorical_n_atoms - 1)
        target_prob = (1 - (atoms_target - self.config.atoms.view(1, -1, 1)).abs() / delta_atom).clamp(0, 1) * \
                      prob_next.unsqueeze(1)
        target_prob = target_prob.sum(-1)
        return target_prob

    def _compute_kl_divergence(self, target_prob, log_prob):
        KL = (target_prob * target_prob.add(1e-5).log() - target_prob * log_prob).sum(-1)
        return KL

    def reduce_loss(self, loss):
        return loss.mean()

    def train_step(self):
        loss = self._compute_loss_and_optimize()
        self.update_step += 1
        self._update_target_network()
        return loss

    def _compute_loss_and_optimize(self):
        losses = []
        for transitions in self.replay.sample():
            loss = self.compute_loss(transitions)
            loss = self.reduce_loss(loss)
            losses.append(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return np.mean(losses)

    def _update_target_network(self):
        if self.update_step % self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
