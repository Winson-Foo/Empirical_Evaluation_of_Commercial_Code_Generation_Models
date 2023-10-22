from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision


class DDPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.actor_network = config.network_fn() # Changed to better reflect what this network does
        self.critic_network = config.network_fn() # Changed to better reflect what this network does
        self.target_actor_network = config.network_fn() # Changed to better reflect what this network does
        self.target_critic_network = config.network_fn() # Changed to better reflect what this network does
        self.target_actor_network.load_state_dict(self.actor_network.state_dict())
        self.target_critic_network.load_state_dict(self.critic_network.state_dict())
        self.replay_buffer = config.replay_fn() # Changed to better reflect what this buffer does
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.current_state = None

    # Soft update the target networks to move them closer to our current networks
    def soft_update_target_networks(self):
        self.soft_update(self.target_actor_network, self.actor_network)
        self.soft_update(self.target_critic_network, self.critic_network)

    # Update the critic network
    def update_critic_network(self):
        transitions = self.replay_buffer.sample()
        states = tensor(transitions.state)
        actions = tensor(transitions.action)
        rewards = tensor(transitions.reward).unsqueeze(-1)
        next_states = tensor(transitions.next_state)
        mask = tensor(transitions.mask).unsqueeze(-1)

        # Compute the expected Q value for the next state
        next_state_features = self.target_actor_network.feature(next_states)
        next_state_actions = self.target_actor_network.actor(next_state_features)
        next_state_q_values = self.target_critic_network(next_state_features, next_state_actions)
        next_state_q_values = self.config.discount * mask * next_state_q_values
        next_state_q_values.add_(rewards)
        next_state_q_values = next_state_q_values.detach()

        # Compute the Q value for the current state and action
        state_features = self.actor_network.feature(states)
        current_q_values = self.critic_network(state_features, actions)

        # Compute the critic loss and update the network
        critic_loss = (current_q_values - next_state_q_values).pow(2).mul(0.5).sum(-1).mean()
        self.critic_network.zero_grad()
        critic_loss.backward()
        self.critic_network.critic_opt.step()

    # Update the actor network
    def update_actor_network(self):
        transitions = self.replay_buffer.sample()
        states = tensor(transitions.state)

        # Compute the actor loss and update the network
        state_features = self.actor_network.feature(states)
        actions = self.actor_network.actor(state_features)
        actor_loss = -self.critic_network(state_features.detach(), actions).mean()
        self.actor_network.zero_grad()
        actor_loss.backward()
        self.actor_network.actor_opt.step()

    # Take a step in the environment
    def step(self):
        if self.current_state is None:
            # Reset the environment and get the initial state
            self.random_process.reset_states()
            self.current_state = self.task.reset()
            self.current_state = self.config.state_normalizer(self.current_state)

        if self.total_steps < self.config.warm_up:
            action = [self.task.action_space.sample()]
        else:
            # Choose an action using the actor network and add some noise
            action = self.actor_network(self.current_state)
            action = to_np(action)
            action += self.random_process.sample()

        # Clip the action to the action space bounds and take a step in the environment
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        reward = self.config.reward_normalizer(reward)
        self.record_online_return(info)

        # Store the transition in the replay buffer
        self.replay_buffer.feed(dict(
            state=self.current_state,
            action=action,
            reward=reward,
            next_state=next_state,
            mask=1-np.asarray(done, dtype=np.int32),
        ))

        if done[0]:
            self.random_process.reset_states()

        # Update the networks if the replay buffer is large enough
        if self.replay_buffer.size() >= self.config.warm_up:
            self.update_critic_network()
            self.update_actor_network()
            self.soft_update_target_networks()

        # Update the current state and total steps taken
        self.current_state = next_state
        self.total_steps += 1

    # Evaluate the agent on a single step
    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.actor_network(state)
        self.config.state_normalizer.unset_read_only()
        return to_np(action)