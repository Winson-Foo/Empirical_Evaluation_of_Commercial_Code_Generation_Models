from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *


class DQNActor(BaseActor):
    """Actor class that performs the interactions between the agent and the environment"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.start()

    def compute_q_values(self, prediction):
        """Computes the Q-values for the current state"""
        q_values = to_np(prediction['q'])
        return q_values

    def transition(self):
        """Performs a transition step from the current state to the next state"""
        if self.current_state is None:
            self.current_state = self.task.reset()

        config = self.config

        if config.noisy_linear:
            self.network.reset_noise()

        with config.lock:
            prediction = self.network(config.state_normalizer(self.current_state))
        
        q_values = self.compute_q_values(prediction)

        if config.noisy_linear:
            epsilon = 0
        elif self.total_steps < config.exploration_steps:
            epsilon = 1
        else:
            epsilon = config.random_action_prob()

        action = epsilon_greedy(epsilon, q_values)
        next_state, reward, done, info = self.task.step(action)

        entry = [self.current_state, action, reward, next_state, done, info]
        self.total_steps += 1
        self.current_state = next_state

        return entry


class DQNAgent(BaseAgent):
    """The main DQN agent class"""

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
        self.total_steps = 0

    def close(self):
        """Closes the replay buffer and actor"""
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        """Takes a step during evaluation mode"""
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)

        q_values = self.network(state)['q']
        action = to_np(q_values.argmax(-1))

        self.config.state_normalizer.unset_read_only()

        return action

    def reduce_loss(self, loss):
        """Reduces the loss"""
        return loss.pow(2).mul(0.5).mean()

    def compute_loss(self, transitions):
        """Computes the loss for a batch of transitions"""
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
        rewards = tensor(transitions.reward)
        q_target = rewards + config.discount ** config.n_step * q_next * masks
        actions = tensor(transitions.action).long()
        q = self.network(states)['q']
        q = q.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = q_target - q

        return loss

    def step(self):
        """Performs a training step"""
        config = self.config

        transitions = self.actor.step()

        for states, actions, rewards, next_states, dones, info in transitions:
            self.record_online_return(info)
            self.total_steps += 1
            self.replay.feed(dict(
                state=np.array([s[-1] if isinstance(s, LazyFrames) else s for s in states]),
                action=actions,
                reward=[config.reward_normalizer(r) for r in rewards],
                mask=1 - np.asarray(dones, dtype=np.int32),
            ))

        if self.total_steps > config.exploration_steps:
            transitions = self.replay.sample()

            if config.noisy_linear:
                self.target_network.reset_noise()
                self.network.reset_noise()

            loss = self.compute_loss(transitions)

            if isinstance(transitions, PrioritizedTransition):
                priorities = loss.abs().add(config.replay_eps).pow(config.replay_alpha)
                idxs = tensor(transitions.idx).long()
                self.replay.update_priorities(zip(to_np(idxs), to_np(priorities)))

                sampling_probs = tensor(transitions.sampling_prob)
                weights = sampling_probs.mul(sampling_probs.size(0)).add(1e-6).pow(-config.replay_beta())
                weights = weights / weights.max()
                loss = loss.mul(weights)

            loss = self.reduce_loss(loss)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)

            with config.lock:
                self.optimizer.step()

        if self.total_steps / config.sgd_update_frequency % \
                config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())