# Refactored code:

# I have refactored the code by making it more modular and readable, following the solid principles.

from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *


class DQNActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()

    def compute_q(self, prediction):
        q_values = to_np(prediction['q'])
        return q_values

    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()

        self._get_prediction_and_q_values()
        self._epsilon_greedy_action_selection()
        entry = self._perform_action_in_task()
        self._update_total_steps_and_state(entry)
        return entry

    def _get_prediction_and_q_values(self):
        config = self.config
        if config.noisy_linear:
            self._network.reset_noise()
        with config.lock:
            prediction = self._network(config.state_normalizer(self._state))
        self.q_values = self.compute_q(prediction)

    def _epsilon_greedy_action_selection(self):
        config = self.config
        if config.noisy_linear:
            epsilon = 0
        elif self._total_steps < config.exploration_steps:
            epsilon = 1
        else:
            epsilon = config.random_action_prob()
        self.action = epsilon_greedy(epsilon, self.q_values)

    def _perform_action_in_task(self):
        next_state, reward, done, info = self._task.step(self.action)
        entry = [self._state, self.action, reward, next_state, done, info]
        return entry

    def _update_total_steps_and_state(self, entry):
        self._total_steps += 1
        self._state = entry[3]


class DQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self._initialize_networks()
        self._initialize_replay_and_actor()

        self.total_steps = 0

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)['q']
        action = to_np(q.argmax(-1))
        self.config.state_normalizer.unset_read_only()
        return action

    def reduce_loss(self, loss):
        return loss.pow(2).mul(0.5).mean()

    def compute_loss(self, transitions):
        config = self.config
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)
        with torch.no_grad():
            q_next = self.target_network(next_states)['q'].detach()
            if self.config.double_q:
                best_actions = torch.argmax(self.network(next_states)['q'], dim=-1)
                q_next = q_next.gather(1, best_actions.unsqueeze(-1)).squeeze(1)
            else:
                q_next = q_next.max(1)[0]
        masks = tensor(transitions.mask)
        rewards = tensor(transitions.reward)
        q_target = rewards + self.config.discount ** config.n_step * q_next * masks
        actions = tensor(transitions.action).long()
        q = self.network(states)['q']
        q = q.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = q_target - q
        return loss

    def step(self):
        self._take_action_and_update_replay_buffer()

        if self.total_steps > self.config.exploration_steps:
            transitions = self.replay.sample()
            if self.config.noisy_linear:
                self.target_network.reset_noise()
                self.network.reset_noise()
            loss = self.compute_loss(transitions)
            if isinstance(transitions, PrioritizedTransition):
                priorities = loss.abs().add(self.config.replay_eps).pow(self.config.replay_alpha)
                idxs = tensor(transitions.idx).long()
                self.replay.update_priorities(zip(to_np(idxs), to_np(priorities)))
                sampling_probs = tensor(transitions.sampling_prob)
                weights = sampling_probs.mul(sampling_probs.size(0)).add(1e-6).pow(-self.config.replay_beta())
                weights = weights / weights.max()
                loss = loss.mul(weights)

            loss = self.reduce_loss(loss)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            with self.config.lock:
                self.optimizer.step()

        if self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def _initialize_networks(self):
        self.network = self.config.network_fn()
        self.network.share_memory()

        self.target_network = self.config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = self.config.optimizer_fn(self.network.parameters())

    def _initialize_replay_and_actor(self):
        self.replay = self.config.replay_fn()

        self.actor = DQNActor(self.config)
        self.actor.set_network(self.network)

    def _take_action_and_update_replay_buffer(self):
        transitions = self.actor.step()
        for states, actions, rewards, next_states, dones, info in transitions:
            self.record_online_return(info)
            self.total_steps += 1
            self.replay.feed(dict(
                state=np.array([s[-1] if isinstance(s, LazyFrames) else s for s in states]),
                action=actions,
                reward=[self.config.reward_normalizer(r) for r in rewards],
                mask=1 - np.asarray(dones, dtype=np.int32),
            ))