#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)
# Permission given to modify the code as long as you keep this
# declaration at the top
#######################################################################

from torch import tensor, no_grad, argmax
from numpy import linspace
from multiprocessing import Lock
from .network import Network
from .component import ReplayBuffer
from .utils import range_tensor
from .BaseAgent import BaseAgent
from .DQN_agent import DQNActor, DQNAgent


class CategoricalDQNActor(DQNActor):
    def __init__(self, config):
        super().__init__(config)

    def _set_up(self):
        self.config.atoms = tensor(self.config.atoms)

    def compute_q(self, prediction):
        q_values = (prediction['prob'] * self.config.atoms).sum(-1)
        return q_values.detach().numpy()


class CategoricalDQNAgent(DQNAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config.lock = Lock()
        self.config.atoms = linspace(
            config.categorical_v_min,
            config.categorical_v_max,
            config.categorical_n_atoms)

        self.replay = ReplayBuffer(
            config.replay_buffer_size,
            config.replay_buffer_type)
        self.actor = CategoricalDQNActor(config)

        self.network = Network(config.state_dim, config.action_dim, config.hidden_dim)
        self.network.share_memory()
        self.target_network = Network(
            config.state_dim,
            config.action_dim,
            config.hidden_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(
            self.network.parameters(),
            lr=config.optimizer_lr)

        self.actor.set_network(self.network)

        self.total_steps = 0
        self.batch_indices = range_tensor(config.batch_size)
        self.atoms = tensor(config.atoms)
        self.delta_atom = (config.categorical_v_max - config.categorical_v_min) \
            / float(config.categorical_n_atoms - 1)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        prediction = self.network(state)
        q = (prediction['prob'] * self.atoms).sum(-1)
        action = argmax(q.detach().numpy(), axis=-1)
        self.config.state_normalizer.unset_read_only()
        return action

    def compute_loss(self, transitions):
        config = self.config
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)
        with no_grad():
            prob_next = self.target_network(next_states)['prob']
            q_next = (prob_next * self.atoms).sum(-1)
            if config.double_q:
                a_next = argmax(
                    (self.network(next_states)['prob'] * self.atoms).sum(-1), dim=-1)
            else:
                a_next = argmax(q_next, dim=-1)
            prob_next = prob_next[self.batch_indices, a_next, :]

        rewards = tensor(transitions.reward).unsqueeze(-1)
        masks = tensor(transitions.mask).unsqueeze(-1)
        atoms_target = rewards \
            + self.config.discount ** config.n_step \
            * masks \
            * self.atoms.view(1, -1)
        atoms_target.clamp_(
            self.config.categorical_v_min,
            self.config.categorical_v_max)
        atoms_target = atoms_target.unsqueeze(1)
        target_prob = (1 - (atoms_target - self.atoms.view(1, -1, 1))
                       .abs() / self.delta_atom).clamp(0, 1) \
            * prob_next.unsqueeze(1)
        target_prob = target_prob.sum(-1)

        log_prob = self.network(states)['log_prob']
        actions = tensor(transitions.action).long()
        log_prob = log_prob[self.batch_indices, actions, :]
        KL = (target_prob * target_prob.add(1e-5).log() - target_prob * log_prob).sum(-1)
        return KL

    def reduce_loss(self, loss):
        return loss.mean()