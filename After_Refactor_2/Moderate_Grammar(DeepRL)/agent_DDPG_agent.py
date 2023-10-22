from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision

# Define constants
HALF = 0.5

class DDPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config

        # Initialize the required components
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None

    # Helper function to update the target network
    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

    def eval_step(self, state):
        # Normalize the state using the state_normalizer
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)

        # Generate an action using the network
        action = self.network(state)

        # Denormalize the action and return it
        action = to_np(action)
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config

        if self.state is None:
            # Reset the state and random process if the current state is None
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = config.state_normalizer(self.state)

        # Choose an action based on the current state, network, and random process
        if self.total_steps < config.warm_up:
            action = [self.task.action_space.sample()]
        else:
            action = self.network(self.state)
            action = to_np(action)
            action += self.random_process.sample()
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)

        # Take a step and record the rewards
        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)

        # Feed the transition to the replay buffer
        self.replay.feed(dict(
            state=self.state,
            action=action,
            reward=reward,
            next_state=next_state,
            mask=1-np.asarray(done, dtype=np.int32),
        ))

        # Reset the random process if the episode is done
        if done[0]:
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        # Update the network using the replay buffer
        if self.replay.size() >= config.warm_up:
            transitions = self.replay.sample()
            states = tensor(transitions.state)
            actions = tensor(transitions.action)
            rewards = tensor(transitions.reward).unsqueeze(-1)
            next_states = tensor(transitions.next_state)
            mask = tensor(transitions.mask).unsqueeze(-1)

            # Update the critic network
            phi_next = self.target_network.feature(next_states)
            a_next = self.target_network.actor(phi_next)
            q_next = self.target_network.critic(phi_next, a_next)
            q_next = config.discount * mask * q_next
            q_next.add_(rewards)
            q_next = q_next.detach()
            phi = self.network.feature(states)
            q = self.network.critic(phi, actions)
            critic_loss = (q - q_next).pow(2).mul(HALF).sum(-1).mean()

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            # Update the actor network
            phi = self.network.feature(states)
            action = self.network.actor(phi)
            policy_loss = -self.network.critic(phi.detach(), action).mean()

            self.network.zero_grad()
            policy_loss.backward()
            self.network.actor_opt.step()

            # Update the target network
            self.soft_update(self.target_network, self.network)