from src.network import *
from src.component import *
import torchvision


class DDPGAgent(BaseAgent):
    """
    Deep Deterministic Policy Gradient agent implementation.

    Args:
        config (Config): configuration object with parameters for the agent.

    Attributes:
        task (Task): task object representing the environment.
        network (DDPGNet): neural network for the agent.
        target_network (DDPGNet): target network for the agent.
        replay_buffer (ReplayBuffer): buffer for storing experience replay data.
        random_process (Exploration): exploration process for the agent.
        total_steps (int): total number of steps taken by the agent.
        state (ndarray): current state of the agent.
    """

    def __init__(self, config):
        super().__init__(config)
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay_buffer = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None

    def soft_update(self):
        """
        Update the target network parameters using a soft update strategy.
        """
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

    def evaluate(self, state):
        """
        Evaluate the network on a given state and return the resulting action.

        Args:
            state (ndarray): input state.

        Returns:
            ndarray: output action.
        """
        self.config.state_normalizer.set_read_only()
        state_ = self.config.state_normalizer(state)
        action = self.network(state_)
        self.config.state_normalizer.unset_read_only()
        return to_np(action)

    def step(self):
        """
        Take a step in the environment and update the agent's parameters.
        """
        config = self.config
        if self.state is None:
            self.random_process.reset()
            self.state = self.task.reset()
            self.state = config.state_normalizer(self.state)

        if self.total_steps < config.warm_up:
            action = [self.task.action_space.sample()]
        else:
            action = self.network(self.state)
            action = to_np(action)
            action += self.random_process.sample()
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)

        self.replay_buffer.feed(dict(
            state=self.state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=1-np.asarray(done, dtype=np.int32),
        ))

        if done[0]:
            self.random_process.reset()
        self.state = next_state
        self.total_steps += 1

        if self.replay_buffer.size() >= config.warm_up:
            transitions = self.replay_buffer.sample()
            states = tensor(transitions.state)
            actions = tensor(transitions.action)
            rewards = tensor(transitions.reward).unsqueeze(-1)
            next_states = tensor(transitions.next_state)
            done = tensor(transitions.done).unsqueeze(-1)

            phi_next = self.target_network.features(next_states)
            a_next = self.target_network.actor(phi_next)
            q_next = self.target_network.critic(phi_next, a_next)
            q_next = config.discount * done * q_next
            q_next.add_(rewards)
            q_next = q_next.detach()

            phi = self.network.features(states)
            q = self.network.critic(phi, actions)
            critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            phi = self.network.features(states)
            action = self.network.actor(phi)
            policy_loss = -self.network.critic(phi.detach(), action).mean()

            self.network.zero_grad()
            policy_loss.backward()
            self.network.actor_opt.step()

            self.soft_update()