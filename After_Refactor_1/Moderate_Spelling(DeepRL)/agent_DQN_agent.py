from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *


class DQNActor(BaseActor):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.start()

    def compute_q_values(self, prediction):
        return to_np(prediction['q'])

    def get_action(self, q_values):
        if self.config.noisy_linear:
            epsilon = 0
        elif self.total_steps < self.config.exploration_steps:
            epsilon = 1
        else:
            epsilon = self.config.random_action_prob()
        return epsilon_greedy(epsilon, q_values)

    def transition(self):
        if self.state is None:
            self.state = self.task.reset()
        if self.config.noisy_linear:
            self.network.reset_noise()
        with self.config.lock:
            prediction = self.network(self.config.state_normalizer(self.state))
        q_values = self.compute_q_values(prediction)
        action = self.get_action(q_values)
        next_state, reward, done, info = self.task.step(action)
        entry = [self.state, action, reward, next_state, done, info]
        self.total_steps += 1
        self.state = next_state
        return entry


class DQNAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = DQNActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def evaluate(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)['q']
        action = to_np(q.argmax(-1))
        self.config.state_normalizer.unset_read_only()
        return action

    def reduce_loss(self, loss):
        return loss.pow(2).mul(0.5).mean()

    def compute_loss(self, transitions):
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
        rewards = tensor(self.config.reward_normalizer(transitions.reward))
        q_target = rewards + self.config.discount ** self.config.n_step * q_next * masks

        actions = tensor(transitions.action).long()
        q = self.network(states)['q'].gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = q_target - q

        if isinstance(transitions, PrioritizedTransition):
            priorities = loss.abs().add(self.config.replay_eps).pow(self.config.replay_alpha)
            indices = tensor(transitions.idx).long()
            self.replay.update_priorities(zip(to_np(indices), to_np(priorities)))

            sampling_probs = tensor(transitions.sampling_prob)
            weights = (sampling_probs * sampling_probs.size(0) + 1e-6).pow(-self.config.replay_beta())
            weights = weights / weights.max()
            loss = loss * weights

        loss = self.reduce_loss(loss)
        return loss

    def train_step(self):
        transitions = self.actor.step()
        for state, action, reward, next_state, done, info in transitions:
            self.record_online_return(info)
            self.total_steps += 1
            self.replay.feed(dict(
                state=np.array([s[-1] if isinstance(s, LazyFrames) else s for s in state]),
                action=action,
                reward=reward,
                mask=1 - np.asarray(done, dtype=np.int32),
            ))

        if self.total_steps > self.config.exploration_steps:
            transitions = self.replay.sample()
            if self.config.noisy_linear:
                self.target_network.reset_noise()
                self.network.reset_noise()
            loss = self.compute_loss(transitions)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)

            with self.config.lock:
                self.optimizer.step()

            if self.total_steps % self.config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())