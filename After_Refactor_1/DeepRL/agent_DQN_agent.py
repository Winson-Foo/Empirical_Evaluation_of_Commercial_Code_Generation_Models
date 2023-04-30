from ..networks import *
from ..components import *
from ..utils import *
from .base_agent import BaseAgent


class DQNAgent(BaseAgent):
    """
    A DQN agent.
    """

    def __init__(self, config):
        """
        Initialize the agent.

        Args:
        - config: a configuration object
        """
        BaseAgent.__init__(self, config)
        self.config = config
        self.replay = config.replay_fn()
        self.actor = DQNActor(config)
        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)

        self.total_steps = 0
        self.sgd_updates = 0

    def close(self):
        """
        Clean up resources used by the agent.
        """
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        """
        Take a step in the environment in evaluation mode.

        Args:
        - state: the current state of the environment

        Returns:
        - An action to take
        """
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)['q']
        action = to_np(q.argmax(-1))
        self.config.state_normalizer.unset_read_only()
        return action

    def reduce_loss(self, loss):
        """
        Reduce the loss tensor to a scalar.

        Args:
        - loss: a tensor of losses

        Returns:
        - A scalar loss
        """
        return loss.pow(2).mul(0.5).mean()

    def compute_loss(self, transitions):
        """
        Compute the loss tensor for a batch of transitions.

        Args:
        - transitions: a batch of transitions

        Returns:
        - A tensor of losses
        """
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

    def train_step(self):
        """
        Take a single training step.
        """
        self.replay.sample()

        if self.config.noisy_linear:
            self.target_network.reset_noise()
            self.network.reset_noise()

        loss = self.compute_loss(self.replay.sample())

        if isinstance(self.replay, PrioritizedTransition):
            priorities = loss.abs().add(self.config.replay_eps).pow(self.config.replay_alpha)
            idxs = tensor(self.replay.idx).long()
            self.replay.update_priorities(zip(to_np(idxs), to_np(priorities)))
            sampling_probs = tensor(self.replay.sampling_prob)
            weights = sampling_probs.mul(sampling_probs.size(0)).add(1e-6).pow(-self.config.replay_beta())
            weights = weights / weights.max()
            loss = loss.mul(weights)

        loss = self.reduce_loss(loss)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)

        with self.config.lock:
            self.optimizer.step()

    def step(self):
        """
        Take a step in the environment and perform a training step if possible.
        """
        transitions = self.actor.step()

        for states, actions, rewards, next_states, dones, info in transitions:
            self.record_online_return(info)

            self.total_steps += 1

            self.replay.feed(dict(
                state=np.array([
                    s[-1] if isinstance(s, LazyFrames) else s for s in states
                ]),
                action=actions,
                reward=[self.config.reward_normalizer(r) for r in rewards],
                mask=1 - np.asarray(dones, dtype=np.int32),
            ))

        if self.total_steps > self.config.exploration_steps and self.total_steps % self.config.sgd_update_frequency == 0:
            self.train_step()

            self.sgd_updates += 1

            if self.sgd_updates % self.config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())