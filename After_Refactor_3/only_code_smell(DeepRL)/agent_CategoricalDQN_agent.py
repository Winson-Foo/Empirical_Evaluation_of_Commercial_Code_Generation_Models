from ..network import *
from ..utils import *
from .BaseAgent import *
from .DQN_agent import *
import torch.multiprocessing as mp

class CategoricalDQNActor(DQNActor):
    def _set_up(self):
        self.atoms = tensor(self.config.atoms)

    def compute_q(self, prediction):
        q_values = torch.matmul(prediction['prob'], self.atoms)
        return to_np(q_values)

class CategoricalDQNAgent(DQNAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config.lock = mp.Lock()
        self._create_atoms()
        self.replay = config.replay_fn()
        self.actor = CategoricalDQNActor(config)
        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.actor.set_network(self.network)

    def _create_atoms(self):
        self.config.atoms = np.linspace(self.config.categorical_v_min,
                                        self.config.categorical_v_max, self.config.categorical_n_atoms)
        self.atoms = tensor(self.config.atoms)
        self.delta_atom = (self.config.categorical_v_max - self.config.categorical_v_min) / float(self.config.categorical_n_atoms - 1)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        prediction = self.network(state)
        q = torch.matmul(prediction['prob'], self.atoms)
        action = to_np(q.argmax(-1))
        self.config.state_normalizer.unset_read_only()
        return action

    def compute_loss(self, transitions):
        config = self.config
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)
        with torch.no_grad():
            prob_next = self.target_network(next_states)['prob']
            q_next = torch.matmul(prob_next, self.atoms)
            if config.double_q:
                a_next = torch.argmax(torch.matmul(self.network(next_states)['prob'], self.atoms), dim=-1)
            else:
                a_next = torch.argmax(q_next, dim=-1)
            prob_next = prob_next[range_tensor(config.batch_size), a_next, :]

        rewards = tensor(transitions.reward).unsqueeze(-1)
        masks = tensor(transitions.mask).unsqueeze(-1)
        atoms_target = rewards + self.config.discount ** config.n_step * masks * self.atoms.view(1, -1)
        atoms_target.clamp_(self.config.categorical_v_min, self.config.categorical_v_max)
        atoms_target = atoms_target.unsqueeze(1)
        target_prob = (1 - (atoms_target - self.atoms.view(1, -1, 1)).abs() / self.delta_atom).clamp(0, 1) * \
                      prob_next.unsqueeze(1)
        target_prob = target_prob.sum(-1)

        log_prob = self.network(states)['log_prob']
        actions = tensor(transitions.action).long()
        log_prob = log_prob[range_tensor(config.batch_size), actions, :]
        KL = (target_prob * target_prob.add(1e-5).log() - target_prob * log_prob).sum(-1)
        return KL

    def reduce_loss(self, loss):
        return loss.mean()