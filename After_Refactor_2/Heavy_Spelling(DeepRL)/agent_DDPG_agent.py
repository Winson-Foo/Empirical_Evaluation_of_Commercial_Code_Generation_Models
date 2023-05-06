from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision


class DDPGAgent(BaseAgent):
    """
    An implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm.

    Args:
        config: A configuration object that includes all necessary parameters.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None

    def soft_update(self, target, src):
        """
        Update the target network by taking a weighted average of its parameters and the source network's parameters.

        Args:
            target: The target network.
            src: The source network.
        """
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

    def eval_step(self, state):
        """
        Get the next action to take in the current state for evaluation purposes.

        Args:
            state: The current state.

        Returns:
            The action to take in the current state.
        """
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return to_np(action)

    def step(self):
        """
        Take a step in the environment and update the network weights if necessary.
        """
        config = self.config
        
        # If this is the first step, reset the random process and normalize the initial state.
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = config.state_normalizer(self.state)

        # Choose an action based on the current state and the network.
        if self.total_steps < config.warm_up:
            action = [self.task.action_space.sample()]
        else:
            action = self.network(self.state)
            action = to_np(action)
            action += self.random_process.sample()
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        
        # Take a step in the environment based on the chosen action.
        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)

        # Add the current state, action, reward, next state, and mask to the replay buffer.
        self.replay.feed(dict(
            state=self.state,
            action=action,
            reward=reward,
            next_state=next_state,
            mask=1-np.asarray(done, dtype=np.int32),
        ))

        # If the episode is done, reset the random process.
        if done[0]:
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        # If the replay buffer is sufficiently full, update the network weights.
        if self.replay.size() >= config.warm_up:
            transitions = self.replay.sample()
            states = tensor(transitions.state)
            actions = tensor(transitions.action)
            rewards = tensor(transitions.reward).unsqueeze(-1)
            next_states = tensor(transitions.next_state)
            mask = tensor(transitions.mask).unsqueeze(-1)

            # Compute the Q values and update the critic network.
            phi_next = self.target_network.feature(next_states)
            a_next = self.target_network.actor(phi_next)
            q_next = self.target_network.critic(phi_next, a_next)
            q_next = config.discount * mask * q_next
            q_next.add_(rewards)
            q_next = q_next.detach()
            phi = self.network.feature(states)
            q = self.network.critic(phi, actions)
            critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            # Compute the policy gradient and update the actor network.
            phi = self.network.feature(states)
            action = self.network.actor(phi)
            policy_loss = -self.network.critic(phi.detach(), action).mean()

            self.network.zero_grad()
            policy_loss.backward()
            self.network.actor_opt.step()

            # Update the target network using a soft update.
            self.soft_update(self.target_network, self.network)