from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *

class DQNActor(BaseActor):
    def __init__(self, config):
        super().__init__(config)
        self.network = config.network_fn()
        self.state = None
        self.total_steps = 0
        self.config = config
        self.epsilon = 1.0

    def transition(self):
        if self.state is None:
            self.state = self.task.reset()

        prediction = self.network(self.config.state_normalizer(self.state))
        q_values = to_np(prediction['q'])

        action = epsilon_greedy(self.epsilon, q_values)
        next_state, reward, done, info = self.task.step(action)
        entry = [self.state, action, reward, next_state, done, info]

        self.total_steps += 1
        self.state = next_state
        if self.total_steps > self.config.exploration_steps:
            self.epsilon = self.config.random_action_prob()

        return entry

class DQNAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.replay = config.replay_fn()
        self.actor = DQNActor(config)

        self.network = None
        self.target_network = None
        self.optimizer = None
        self.init_network()

        self.total_steps = 0

    def init_network(self):
        self.network = config.network_fn()
        self.network.share_memory()

        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_task(self.task)
        self.actor.set_network(self.network)

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
        rewards = tensor(transitions.reward)

        q_target = rewards + self.config.discount ** self.config.n_step * q_next * masks
        actions = tensor(transitions.action).long()
        q = self.network(states)['q']
        q = q.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        loss = q_target - q
        return loss

    def reduce_loss(self, loss):
        return loss.pow(2).mul(0.5).mean()

    def step(self):
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

        if self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        if self.total_steps > self.config.exploration_steps:
            transitions = self.replay.sample()

            if isinstance(transitions, PrioritizedTransition):
                priorities = self.compute_loss(transitions)
                priorities = priorities.abs().add(self.config.replay_eps).pow(self.config.replay_alpha)
                idxs = tensor(transitions.idx).long()
                self.replay.update_priorities(zip(to_np(idxs), to_np(priorities)))
                sampling_probs = tensor(transitions.sampling_prob)
                weights = sampling_probs.mul(sampling_probs.size(0)).add(1e-6).pow(-self.config.replay_beta())
                weights = weights / weights.max()
                loss = priorities.mul(weights)
            else:
                loss = self.compute_loss(transitions)

            loss = self.reduce_loss(loss)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            self.optimizer.step()