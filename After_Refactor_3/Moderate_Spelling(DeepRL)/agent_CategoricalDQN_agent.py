from typing import List
import torch.multiprocessing as mp
from torch import tensor
from ..network import Network
from ..component import ReplayBuffer
from ..utils import to_np, range_tensor
from .BaseAgent import BaseAgent
from .DQN_agent import DQNAgent


class CategoricalDQNActor(DQNAgent):
    """
    Actor for Categorical DQN algorithm
    """

    def __init__(self, network: Network, config: dict):
        super(CategoricalDQNActor, self).__init__(config)
        self.network = network
        self.config = config
        self.atoms = tensor(config['atoms'])

    def compute_state_action_values(self, state: tensor) -> tensor:
        """
        Compute state-action values using the network output
        """
        with torch.no_grad():
            prediction = self.network(state)
            state_action_values = (prediction['prob'] * self.atoms).sum(-1)
        return state_action_values


class CategoricalDQNAgent(BaseAgent):
    """
    Categorical DQN agent
    """

    def __init__(self, config: dict):
        super(CategoricalDQNAgent, self).__init__(config)
        self.config = config
        self.actor: CategoricalDQNActor = None
        self.network: Network = None
        self.target_network: Network = None
        self.optimizer = None
        self.batch_indices = range_tensor(config['batch_size'])
        self.atoms = tensor(config['atoms'])
        self.delta_atom = (config['categorical_v_max'] - config['categorical_v_min']) / float(
            config['categorical_n_atoms'] - 1)
        self.replay: ReplayBuffer = None
        self.total_steps = 0

    def _set_up(self):
        self.config['atoms'] = np.linspace(self.config['categorical_v_min'],
                                           self.config['categorical_v_max'], self.config['categorical_n_atoms'])

        self.replay = self.config['replay_fn']()
        self.network = self.config['network_fn']()
        self.network.share_memory()
        self.target_network = self.config['network_fn']()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = self.config['optimizer_fn'](self.network.parameters())
        self.actor = CategoricalDQNActor(self.network, self.config)

    def eval_step(self, state: tensor) -> int:
        """
        Choose an action for evaluation purposes
        """
        with torch.no_grad():
            self.config['state_normalizer'].set_read_only()
            state = self.config['state_normalizer'](state)
            state_action_values = self.actor.compute_state_action_values(state)
            action = to_np(state_action_values.argmax(-1))
            self.config['state_normalizer'].unset_read_only()
        return action

    def compute_loss(self, transitions: dict) -> tensor:
        """
        Compute the categorical loss
        """
        states, actions, rewards, next_states, masks = transitions['state'], transitions['action'], \
                                                       transitions['reward'], transitions['next_state'], \
                                                       transitions['mask']

        states = self.config['state_normalizer'](states)
        next_states = self.config['state_normalizer'](next_states)

        with torch.no_grad():
            target_logits = self.target_network(next_states)['prob']

            if self.config['double_q']:
                q_values = self.network(next_states)['prob'] * self.atoms
                a_next = q_values.sum(-1).argmax(-1)
                target_probs = target_logits[range(len(next_states)), a_next, :]
            else:
                target_q_values = (target_logits * self.atoms).sum(-1)
                a_next = target_q_values.argmax(-1)
                target_probs = target_logits[range(len(next_states)), a_next, :]

        rewards = tensor(rewards).unsqueeze(-1)
        masks = tensor(masks).unsqueeze(-1)
        atoms_target = rewards + self.config['discount'] ** self.config['n_step'] * masks * self.atoms.view(1, -1)
        atoms_target.clamp_(self.config['categorical_v_min'], self.config['categorical_v_max'])
        atoms_target_idx = (atoms_target - self.config['categorical_v_min']) / self.delta_atom
        atoms_target_idx_floor = atoms_target_idx.floor().long()
        atoms_target_idx_ceil = atoms_target_idx.ceil().long()
        delta = atoms_target - self.atoms[atoms_target_idx_floor]
        target_probs[range(len(next_states)), atoms_target_idx_floor.view(-1), :] += \
            delta * (target_probs[range(len(next_states)), atoms_target_idx_ceil.view(-1), :]
                     - target_probs[range(len(next_states)), atoms_target_idx_floor.view(-1), :])
        target_probs = target_probs.sum(-1)

        logits = self.network(states)['prob']
        log_probs = torch.log(logits[range(len(states)), actions, :])
        kl_divergence = (target_probs * (torch.log(target_probs + 1e-8) - log_probs)).sum(-1)
        return kl_divergence

    def reduce_loss(self, loss: List[tensor]) -> tensor:
        """
        Reduce a list of losses to a single tensor
        """
        return torch.stack(loss).mean()