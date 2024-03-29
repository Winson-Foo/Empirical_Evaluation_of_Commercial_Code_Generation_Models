from ..network import *
from ..utils import *
from .BaseAgent import BaseAgent
from .DQN_agent import DQNAgent


class CategoricalDQNActor(DQNAgent):
    def __init__(self, config):
        super().__init__(config)
        self.atoms = tensor(self.config.atoms)

    def compute_q_values(self, prediction):
        q_values = (prediction['prob'] * self.atoms).sum(-1)
        return to_np(q_values)


class CategoricalDQNAgent(DQNAgent):
    def __init__(self, config):
        super().__init__(config)
        config.lock = mp.Lock()
        config.atoms = np.linspace(config.categorical_v_min, config.categorical_v_max, config.categorical_n_atoms)
        self.atoms = tensor(config.atoms)
        self.delta_atom = (config.categorical_v_max - config.categorical_v_min) / float(config.categorical_n_atoms - 1)
        self.replay = config.replay_fn()
        self.actor = CategoricalDQNActor(config)
        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.actor.set_network(self.network)
        self.total_steps = 0
        self.batch_indices = range_tensor(config.batch_size)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        prediction = self.network(state)
        q_values = self.actor.compute_q_values(prediction)
        action = to_np(q_values.argmax(-1))
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
        atoms_target = rewards + self.config.discount ** self.config.n_step * masks * self.atoms.view(1, -1)
        atoms_target.clamp_(self.config.categorical_v_min, self.config.categorical_v_max)
        atoms_target = atoms_target.unsqueeze(1)
        target_prob = (1 - (atoms_target - self.atoms.view(1, -1, 1)).abs() / self.delta_atom).clamp(0, 1) * \
                      prob_next.unsqueeze(1)
        target_prob = target_prob.sum(-1)

        log_prob = self.network(states)['log_prob']
        actions = tensor(transitions.action).long()
        log_prob = log_prob[self.batch_indices, actions, :]
        kl_divergence = (target_prob * target_prob.add(1e-5).log() - target_prob * log_prob).sum(-1)
        return kl_divergence

    def reduce_loss(self, loss):
        return loss.mean()