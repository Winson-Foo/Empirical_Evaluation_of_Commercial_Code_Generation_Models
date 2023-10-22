# Import necessary packages
import time
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from ..component import *
from ..network import *
from ..utils import *
from .BaseAgent import *

# Define DQNActor class
class DQNActor(BaseActor):
    def __init__(self, config):
        super(DQNActor, self).__init__(config)
        self.config = config
        self.start()

    def compute_q_values(self, prediction):
        q_values = to_np(prediction['q'])
        return q_values

    def _transition(self):
        """
        Plays a single step of the game and returns the resulting state, action, reward, and next state.
        """
        if self.current_state is None:
            self.current_state = self.game.reset()
        config = self.config
        if config.noisy_linear:
            self.network.reset_noise()
        with config.lock:
            prediction = self.network(config.normalize_state(self.current_state))
        q_values = self.compute_q_values(prediction)

        if config.noisy_linear:
            epsilon = 0
        elif self.total_steps < config.exploration_steps:
            epsilon = 1
        else:
            epsilon = config.random_action_prob()
        action = epsilon_greedy(epsilon, q_values)
        next_state, reward, done, info = self.game.step(action)
        transition = [self.current_state, action, reward, next_state, done, info]
        self.total_steps += 1
        self.current_state = next_state
        return transition

# Define DQNAgent class
class DQNAgent(BaseAgent):
    def __init__(self, config):
        super(DQNAgent, self).__init__(config)
        self.config = config

        # Create replay buffer
        self.replay_buffer = config.replay_fn()

        # Create actor and set network
        self.actor = DQNActor(config)
        self.actor.set_network(self.network)

        # Create networks and optimizer
        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.total_steps = 0

    def close(self):
        close_obj(self.replay_buffer)
        close_obj(self.actor)

    def eval_step(self, state):
        """
        Makes a prediction based on the given state and returns the corresponding action.
        """
        self.config.normalize_state.set_read_only()
        state = self.config.normalize_state(state)
        q = self.network(state)['q']
        action = to_np(q.argmax(-1))
        self.config.normalize_state.unset_read_only()
        return action

    def compute_loss(self, batch):
        """
        Computes the loss for the given batch of transitions and returns the resulting loss tensor.
        """
        config = self.config
        states = self.config.normalize_state(batch.state)
        next_states = self.config.normalize_state(batch.next_state)
        with torch.no_grad():
            q_next = self.target_network(next_states)['q'].detach()
            if self.config.double_q:
                best_actions = torch.argmax(self.network(next_states)['q'], dim=-1)
                q_next = q_next.gather(1, best_actions.unsqueeze(-1)).squeeze(1)
            else:
                q_next = q_next.max(1)[0]
        masks = tensor(batch.mask)
        rewards = tensor(batch.reward)
        q_target = rewards + self.config.discount ** config.n_step * q_next * masks
        actions = tensor(batch.action).long()
        q = self.network(states)['q']
        q = q.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = q_target - q
        return loss

    def step(self):
        """
        Samples from the replay buffer, computes the loss, and updates the network weights.
        """
        config = self.config
        transitions = self.actor.step()
        for states, actions, rewards, next_states, dones, info in transitions:
            self.record_online_return(info)
            self.total_steps += 1
            self.replay_buffer.feed(dict(
                state=np.array([s[-1] if isinstance(s, LazyFrames) else s for s in states]),
                action=actions,
                reward=[config.normalize_reward(r) for r in rewards],
                mask=1 - np.asarray(dones, dtype=np.int32),
            ))

        if self.total_steps > self.config.exploration_steps:
            batch = self.replay_buffer.sample()
            if config.noisy_linear:
                self.target_network.reset_noise()
                self.network.reset_noise()
            loss = self.compute_loss(batch)
            if isinstance(batch, PrioritizedTransition):
                priorities = loss.abs().add(config.replay_eps).pow(config.replay_alpha)
                idxs = tensor(batch.idx).long()
                self.replay_buffer.update_priorities(zip(to_np(idxs), to_np(priorities)))
                sampling_probs = tensor(batch.sampling_prob)
                weights = sampling_probs.mul(sampling_probs.size(0)).add(1e-6).pow(-config.replay_beta())
                weights = weights / weights.max()
                loss = loss.mul(weights)

            loss = loss.pow(2).mul(0.5).mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            with config.lock:
                self.optimizer.step()

        if self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())