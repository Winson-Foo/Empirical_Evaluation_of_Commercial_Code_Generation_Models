from ..network import *
from ..component import *
from ..utils import *
import time

class DQNActor:
    def __init__(self, config):
        self.config = config
        self.task = config.task
        self.state = None
        self.network = None
        self.total_steps = 0
        self.start()

    def start(self):
        self.state = self.task.reset()
        self.network = self.config.network_fn()

    def compute_q(self, prediction):
        q_values = to_np(prediction['q'])
        return q_values

    def choose_action(self):
        config = self.config
        prediction = self.network(config.state_normalizer(self.state))
        q_values = self.compute_q(prediction)
        epsilon = config.random_action_prob(self.total_steps)
        action = epsilon_greedy(epsilon, q_values)
        return action, prediction

    def step(self):
        action, prediction = self.choose_action()
        next_state, reward, done, info = self.task.step(action)
        entry = [self.state, action, reward, next_state, done, info]
        self.total_steps += 1
        self.state = next_state
        return entry, prediction


class DQNAgent:
    def __init__(self, config):
        self.config = config
        self.replay = config.replay_fn()
        self.actor = DQNActor(config)
        self.network = self.config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)['q']
        action = to_np(q.argmax(-1))
        self.config.state_normalizer.unset_read_only()
        return action

    def compute_loss(self, transitions):
        config = self.config
        states = config.state_normalizer(transitions.state)
        next_states = config.state_normalizer(transitions.next_state)
        with torch.no_grad():
            q_next = self.target_network(next_states)['q'].detach()
            if config.double_q:
                best_actions = torch.argmax(self.network(next_states)['q'], dim=-1)
                q_next = q_next.gather(1, best_actions.unsqueeze(-1)).squeeze(1)
            else:
                q_next = q_next.max(1)[0]
        masks = tensor(transitions.mask)
        rewards = config.reward_normalizer(tensor(transitions.reward))
        q_target = rewards + config.discount ** config.n_step * q_next * masks
        actions = tensor(transitions.action).long()
        q = self.network(states)['q']
        q = q.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = q_target - q
        return loss

    def update(self, transitions):
        if self.total_steps > self.config.exploration_steps:
            if isinstance(transitions, PrioritizedTransition):
                priorities = self.compute_priorities(transitions)
                self.replay.update_priorities(priorities)
                weights = self.compute_weights(transitions)
                loss = self.reduce_loss(weights.mul(transitions.error))
            else:
                loss = self.reduce_loss(self.compute_loss(transitions))
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            with self.config.lock:
                self.optimizer.step()
            if self.total_steps / self.config.sgd_update_frequency % \
                    self.config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

    def compute_priorities(self, transitions):
        priorities = transitions.error.abs().add(self.config.replay_eps).pow(self.config.replay_alpha)
        return zip(to_np(transitions.idx), to_np(priorities))

    def compute_weights(self, transitions):
        sampling_probs = tensor(transitions.sampling_prob)
        weights = sampling_probs.mul(sampling_probs.size(0)).add(1e-6).pow(-self.config.replay_beta())
        weights = weights / weights.max()
        return weights

    def reduce_loss(self, loss):
        return loss.pow(2).mul(0.5).mean()

    def step(self):
        transitions = []
        for _ in range(self.config.num_steps):
            entry, prediction = self.actor.step()
            self.record_online_return(entry[-1])
            self.replay.feed(dict(
                state=np.array([s[-1] if isinstance(s, LazyFrames) else s for s in entry[0]]),
                action=entry[1],
                reward=[self.config.reward_normalizer(r) for r in entry[2]],
                next_state=np.array([s[-1] if isinstance(s, LazyFrames) else s for s in entry[3]]),
                mask=1 - np.asarray(entry[4], dtype=np.int32),
            ))
            self.total_steps += 1
            if entry[4]:
                break
        if len(self.replay) > self.config.batch_size:
            transitions = self.replay.sample(self.config.batch_size)

        self.update(transitions)