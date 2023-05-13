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

    def compute_q_values(self, prediction):
        q_values = to_np(prediction['q'])
        return q_values

    def _transition(self):
        if self.current_state is None:
            self.current_state = self.task.reset()

        if self.config.noisy_linear:
            self.network.reset_noise()

        with self.config.lock:
            prediction = self.network(self.config.state_normalizer(self.current_state))

        q_values = self.compute_q_values(prediction)

        if self.config.noisy_linear:
            epsilon = 0
        elif self.total_steps < self.config.exploration_steps:
            epsilon = 1
        else:
            epsilon = self.compute_epsilon()

        action = self.compute_next_action(epsilon, q_values)
        next_state, reward, is_done, info = self.task.step(action)
        entry = [self.current_state, action, reward, next_state, is_done, info]

        self.total_steps += 1
        self.current_state = next_state
        return entry

    def compute_epsilon(self):
        return self.config.random_action_prob()

    def compute_next_action(self, epsilon, q_values):
        return epsilon_greedy(epsilon, q_values)


class DQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
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
        self.total_steps = 0

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q_values = self.network(state)['q']
        action = to_np(q_values.argmax(-1))
        self.config.state_normalizer.unset_read_only()
        return action

    def reduce_loss(self, loss):
        return loss.pow(2).mul(0.5).mean()

    def compute_loss(self, transitions):
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)

        with torch.no_grad(), self.target_network:
            q_next = self.target_network(next_states)['q'].detach()
            if self.config.double_q:
                best_actions = torch.argmax(self.network(next_states)['q'], dim=-1)
                q_next = q_next.gather(1, best_actions.unsqueeze(-1)).squeeze(1)
            else:
                q_next = q_next.max(1)[0]

        done_masks = tensor(transitions.mask)
        rewards = tensor(transitions.reward)
        q_targets = rewards + self.config.discount ** self.config.n_step * q_next * done_masks
        actions = tensor(transitions.action).long()
        q_values = self.network(states)['q']
        q = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = q_targets - q
        return loss

    def step(self):
        transitions = self.actor.step()
        for state, action, reward, next_state, is_done, info in transitions:
            self.record_online_return(info)
            self.total_steps += 1
            state = [s[-1] if isinstance(s, LazyFrames) else s for s in state]
            self.replay.feed(dict(
                state=np.array(state),
                action=action,
                reward=[self.config.reward_normalizer(r) for r in reward],
                mask=1 - np.asarray(is_done, dtype=np.int32),
            ))

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