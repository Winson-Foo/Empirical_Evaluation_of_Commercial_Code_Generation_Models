from typing import Dict
import torch.optim as optim
import torch.nn.functional as F

from ..network import Network
from ..component import Replay, RandomProcess, Normalizer
from .BaseAgent import BaseAgent


class TD3Agent(BaseAgent):
    """
    Twin Delayed DDPG (TD3) agent implementation.
    """
    def __init__(self, config: Dict):
        """
        Initializes the TD3Agent.

        Args:
            config (dict): Configurations dictionary.
        """
        super().__init__(config)
        self.task = config['task_fn']()
        self.replay = config['replay_fn']()
        self.random_process = config['random_process_fn']()
        self.critic_opt = optim.Adam(self.network.get_critic_params(), lr=config['critic_lr'])
        self.actor_opt = optim.Adam(self.network.get_actor_params(), lr=config['actor_lr'])
        self.state_normalizer = config['state_normalizer_fn']()
        self.reward_normalizer = config['reward_normalizer_fn']()
        self.total_steps = 0
        self._state = None

    def eval_step(self, state):
        """
        Runs an evaluation step to get the action given the current state.

        Args:
            state (np.ndarray): Environment state.

        Returns:
            np.ndarray: Action to be taken in the given state.
        """
        self.state_normalizer.set_read_only()
        state = self.state_normalizer(state)
        action = self.network.eval(state)
        self.state_normalizer.unset_read_only()
        return action

    def step(self):
        """
        Runs a single step of the TD3 training loop.
        """
        if self._state is None:
            self.random_process.reset_states()
            self._state = self.task.reset()
            self._state = self.state_normalizer(self._state)

        if self.total_steps < self.config['warm_up']:
            action = [self.task.action_space.sample()]
        else:
            action = self.network.train(self._state, self.random_process.sample())
            action = np.clip(action, self.task.action_space.low, self.task.action_space.high)

        next_state, reward, done, info = self.task.step(action)
        next_state = self.state_normalizer(next_state)
        self.record_online_return(info)
        reward = self.reward_normalizer(reward)

        self.replay.feed({'state': self._state,
                          'action': action,
                          'reward': reward,
                          'next_state': next_state,
                          'mask': 1 - done.astype(int)})

        if done[0]:
            self.random_process.reset_states()
        self._state = next_state
        self.total_steps += 1

        if self.total_steps >= self.config['warm_up'] and self.total_steps % self.config['update_interval'] == 0:
            transitions = self.replay.sample()
            self.learn(transitions)

    def learn(self, transitions):
        """
        Runs the TD3 training algorithm.

        Args:
            transitions (dict): Dictionary containing the sampled batch of transitions.
        """
        states = self.state_normalizer(transitions['state'])
        actions = transitions['action']
        rewards = transitions['reward'].unsqueeze(-1)
        next_states = self.state_normalizer(transitions['next_state'])
        masks = transitions['mask'].unsqueeze(-1)

        # Twin Q-networks
        a_next = self.target_network.actor(next_states)
        noise = torch.randn_like(a_next) * self.config['td3_noise']
        noise = noise.clamp(-self.config['td3_noise_clip'], self.config['td3_noise_clip'])
        a_next = (a_next + noise).clamp(self.task.action_space.low.item(), self.task.action_space.high.item())
        q1_next, q2_next = self.target_network.critic(next_states, a_next)
        q_next = torch.min(q1_next, q2_next)

        target = rewards + self.config['discount'] * masks * q_next.detach()
        q1, q2 = self.network.critic(states, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        if self.total_steps % self.config['policy_update_interval'] == 0:
            actor_loss = -self.network.critic(states, self.network.actor(states))[0].mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self.target_network.soft_update(self.network, self.config['tau'])

        self.record_loss('critic_loss', critic_loss.item())
        self.record_loss('actor_loss', actor_loss.item())


class TD3Network(Network):
    """
    Twin Delayed DDPG (TD3) network implementation.
    """
    def __init__(self, input_dim, output_dim, config):
        """
        Initializes the network.

        Args:
            input_dim (tuple): Tuple representing the input dimensions.
            output_dim (tuple): Tuple representing the output dimensions.
            config (dict): Configurations dictionary.
        """
        super().__init__(input_dim, output_dim, config)

    def init_layers(self, config):
        """
        Initializes the layers of the network.

        Args:
            config (dict): Configurations dictionary.

        Returns:
            torch.nn.ModuleList: List of network layers.
        """
        actor_hidden_dim = config['actor_hidden_dim']
        critic_hidden_dim = config['critic_hidden_dim']

        layers = nn.ModuleList()
        layers.append(nn.Linear(self.input_dim, actor_hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(actor_hidden_dim, actor_hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(actor_hidden_dim, self.output_dim))
        layers.append(nn.Tanh())

        layers.append(nn.Linear(self.input_dim + self.output_dim, critic_hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(critic_hidden_dim, critic_hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(critic_hidden_dim, 1))

        layers.append(nn.Linear(self.input_dim + self.output_dim, critic_hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(critic_hidden_dim, critic_hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(critic_hidden_dim, 1))
        return layers

    def forward(self, x):
        """
        Runs a forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        state, action = x
        actor = self.actor(state)
        critic_1, critic_2 = self.critic(state, action)
        return actor, critic_1, critic_2

    def get_critic_params(self):
        """
        Returns the parameters of the critic networks.

        Returns:
            torch.optim.Optimizer: Critic network parameters.
        """
        return list(self.critic[0].parameters()) + \
               list(self.critic[2].parameters()) + \
               list(self.critic[4].parameters())

    def get_actor_params(self):
        """
        Returns the parameters of the actor network.

        Returns:
            torch.optim.Optimizer: Actor network parameters.
        """
        return self.actor.parameters()


def state_normalizer_fn():
    return Normalizer(0)


def reward_normalizer_fn():
    return Normalizer(0)


def replay_fn():
    return Replay(memory_size=10000, batch_size=128)


def random_process_fn():
    return RandomProcess()

def actor_hidden_dim():
    return 256


def critic_hidden_dim():
    return 256


def max_action():
    return 1.0


def warm_up():
    return 10000


def discount_factor():
    return 0.99


def critic_lr():
    return 1e-3


def actor_lr():
    return 1e-4


def td3_noise():
    return 0.2


def td3_noise_clip():
    return 0.5


def target_network_mix():
    return 0.005


def update_interval():
    return 1


def policy_update_interval():
    return 2


def tau():
    return 0.005

config = {
    'task_fn': ...,
    'state_normalizer_fn': state_normalizer_fn,
    'reward_normalizer_fn': reward_normalizer_fn,
    'replay_fn': replay_fn,
    'random_process_fn': random_process_fn,
    'network_fn': TD3Network,
    'actor_hidden_dim': actor_hidden_dim(),
    'critic_hidden_dim': critic_hidden_dim(),
    'max_action': max_action(),
    'warm_up': warm_up(),
    'discount': discount_factor(),
    'critic_lr': critic_lr(),
    'actor_lr': actor_lr(),
    'td3_noise': td3_noise(),
    'td3_noise_clip': td3_noise_clip(),
    'target_network_mix': target_network_mix(),
    'update_interval': update_interval(),
    'policy_update_interval': policy_update_interval(),
    'tau': tau(),
}

agent = TD3Agent(config)