from torch import tensor
from numpy import clip, asarray, int32
from rlkit.network import QNetwork, Actor, Critic
from rlkit.random_process import OrnsteinUhlenbeckProcess
from rlkit.replay_buffer import ReplayBuffer
from rlkit.agent.base import BaseAgent


class DDPGAgent(BaseAgent):
    """
    Deep Deterministic Policy Gradients (DDPG) agent implementation.
    """

    def __init__(self, config):
        """
        :param config: (dict) Configuration parameters.
        """
        self.config = config
        self.task = config.task_fn()
        self.network = QNetwork(config)
        self.target_network = QNetwork(config)
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay_buffer = ReplayBuffer(config.replay_size)
        self.random_process = OrnsteinUhlenbeckProcess(config.action_dim, config.rand_process_theta, \
                                                        config.rand_process_sigma, config.action_range)
        self.total_steps = 0
        self.state = None

    def eval_step(self, state):
        """
        Evaluate the agent's policy at the given state.

        :param state: (array_like) The current state.
        :returns: (array_like) The action to take.
        """
        with torch.no_grad():
            state = self.config.state_normalizer(state)
            action = self.network(state)
            action *= self.config.action_range
            self.config.state_normalizer.unset_read_only()
            return action.cpu().numpy()

    def step(self):
        """
        Run one step of the agent.

        :returns: (float) The episode reward.
        """

        # Reset state and random process when necessary
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = self.config.state_normalizer(self.state)

        # Determine action
        if self.total_steps < self.config.warm_up:
            action = [self.task.action_space.sample()]
        else:
            action = self.network(self.state)
            action *= self.config.action_range
            action += self.random_process.sample()
        action = clip(action, self.task.action_space.low, self.task.action_space.high)

        # Update state, record return, and normalize reward
        next_state, reward, done, _ = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)
        done = 1 - asarray(done, dtype=int32)

        # Add to replay buffer
        self.replay_buffer.add(self.state, action, reward, next_state, done)

        # Reset random process if the environment is done
        if done[0]:
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        # Train the network once the replay buffer is full enough
        if self.replay_buffer.size() >= self.config.warm_up:
            transitions = self.replay_buffer.sample()
            states = tensor(transitions.state)
            actions = tensor(transitions.action)
            rewards = tensor(transitions.reward)
            next_states = tensor(transitions.next_state)
            masks = tensor(transitions.mask)

            # Compute critic and actor losses
            phi_next = self.target_network.feature(next_states)
            a_next = self.target_network.actor(phi_next)
            q_next = self.target_network.critic(phi_next, a_next)
            q_next *= self.config.discount * masks
            q_next += rewards
            q_next = q_next.detach()
            phi = self.network.feature(states)
            q = self.network.critic(phi, actions)
            critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            phi = self.network.feature(states)
            action = self.network.actor(phi)
            policy_loss = -self.network.critic(phi.detach(), action).mean()

            self.network.zero_grad()
            policy_loss.backward()
            self.network.actor_opt.step()

            self.soft_update(self.target_network, self.network)