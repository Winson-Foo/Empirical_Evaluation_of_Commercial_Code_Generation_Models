from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *
import torch.nn.functional as F


class DQNActor(BaseActor):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.start()

    def compute_q(self, prediction):
        q_values = to_np(prediction['q'])
        return q_values

    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()

        if self.config.noisy_linear:
            self._network.reset_noise()

        with self.config.lock:
            prediction = self._network(self.config.state_normalizer(self._state))

        q_values = self.compute_q(prediction)

        if self.config.noisy_linear:
            epsilon = 0
        elif self._total_steps < self.config.exploration_steps:
            epsilon = 1
        else:
            epsilon = self.config.random_action_prob()

        action = epsilon_greedy(epsilon, q_values)

        next_state, reward, done, info = self._task.step(action)

        entry = [self._state, action, reward, next_state, done, info]

        self._total_steps += 1
        self._state = next_state

        return entry


class DQNAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.config.lock = mp.Lock()

        self.replay_buffer = ReplayBuffer(config.replay_size, config.batch_size)

        self.network = DQNNetwork(config.state_dim, config.action_dim, config.hidden_dim)
        self.network.share_memory()

        self.target_network = DQNNetwork(config.state_dim, config.action_dim, config.hidden_dim)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)

        self.actor = DQNActor(config)
        self.actor.set_network(self.network)

        self.total_steps = 0

    def close(self):
        close_obj(self.replay_buffer)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)['q']
        action = to_np(q.argmax(-1))
        self.config.state_normalizer.unset_read_only()
        return action

    def compute_loss(self, transitions):
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)

        with torch.no_grad():
            next_q = self.target_network(next_states)['q']

            if self.config.double_q:
                best_actions = torch.argmax(self.network(next_states)['q'], dim=-1)
                next_q = next_q.gather(1, best_actions.unsqueeze(-1)).squeeze(1)
            else:
                next_q = next_q.max(1)[0]

        masks = transitions.mask
        rewards = transitions.reward
        q_targets = rewards + self.config.gamma * next_q * masks

        actions = transitions.action
        q_values = self.network(states)['q']
        q = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        loss = F.mse_loss(q, q_targets)

        return loss

    def train_step(self):
        transitions = self.replay_buffer.sample()

        loss = self.compute_loss(transitions)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()

        priorities = torch.abs(loss).add(1e-6).pow(self.config.alpha)
        self.replay_buffer.update_priorities(transitions.indices, priorities)

    def step(self):
        transitions = self.actor.step()

        for state, action, reward, next_state, done, info in transitions:
            self.record_online_return(info)

            self.total_steps += 1

            self.replay_buffer.add(state, action, reward, next_state, done)

        if self.total_steps > self.config.exploration_steps:
            if self.config.noisy_linear:
                self.target_network.reset_noise()
                self.network.reset_noise()

            self.train_step()

            if self.total_steps / self.config.sgd_update_frequency % self.config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())