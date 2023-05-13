import numpy as np
import torch
import time
import os

from .experience_replay import ReplayMemory
from .environment import Environment

from ..utils.torch_utils import *
from ..utils.misc import *


class Agent:
    def __init__(self, config):
        """
        Represents a reinforcement learning agent.

        :param config: Configuration dictionary.
        """
        self.config = config
        self.total_steps = 0
        generate_tag(config)
        self.log_dir = config.get('log_dir', get_default_log_dir(config['tag']))
        self.logger = getLogger(self.log_dir)
        self.logger.info(config)
        self.memory = ReplayMemory(config['memory_size'])
        self.env = Environment(config)
        self.net = config['network_fn'](self.env.state_shape, self.env.num_actions)
        self.optimizer = torch.optim.Adam(self.net.parameters(), config['learning_rate'])
        self.target_net = config['network_fn'](self.env.state_shape, self.env.num_actions)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()
        if torch.cuda.is_available():
            self.net.cuda()
            self.target_net.cuda()
        self.steps_per_task = config.get('steps_per_task', float('inf'))
        self.reset()

    def reset(self):
        """
        Resets the agent's state.
        """
        self.env.reset()
        self.state = self.env.get_state()
        self.total_reward = 0.0
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.episode_num = 0

    def step(self):
        """
        Runs a step of the agent's training procedure.
        """
        self.total_steps += 1
        self.episode_steps += 1
        epsilon = self.config['epsilon_by_step'](self.total_steps)
        action = self.net.act(to_var(self.state, requires_grad=False), epsilon)
        next_state, reward, done = self.env.step(action)
        self.total_reward += reward
        self.episode_reward += reward
        self.memory.push(self.state, action, reward, next_state, done)
        self.state = next_state
        if len(self.memory) >= self.config['batch_size'] and self.total_steps % self.config['train_frequency'] == 0:
            transitions = self.memory.sample(self.config['batch_size'])
            batch = Transition(*zip(*transitions))
            loss = self.net.train_model(self.target_net, self.optimizer, batch, self.config['gamma'])
            self.logger.info(f'step {self.total_steps}, loss {loss:.5f}')
        if self.total_steps % self.config['target_update_frequency'] == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        if done:
            self.episode_num += 1
            self.logger.info(f'step {self.total_steps}, episode {self.episode_num}, reward {self.episode_reward:.1f}')
            self.episode_reward = 0.0
            self.reset()

    def switch_task(self):
        """
        Switches the agent's current active task if necessary.
        """
        if self.env.intrinsic_reward > self.config['task_threshold'] and self.episode_steps >= self.steps_per_task:
            self.episode_steps = 0
            self.env.switch_task()

    def eval(self):
        """
        Evaluates the agent's performance in the environment.
        """
        self.net.eval()
        self.reset()
        while self.total_steps < self.config['eval_steps']:
            epsilon = self.config['epsilon_by_step'](self.total_steps)
            action = self.net.act(to_var(self.state, requires_grad=False), epsilon)
            next_state, reward, done = self.env.step(action, True)
            self.total_reward += reward
            self.episode_reward += reward
            self.state = next_state
            if done:
                self.episode_num += 1
                self.logger.info(f'step {self.total_steps}, episode {self.episode_num}, reward {self.episode_reward:.1f}')
                self.episode_reward = 0.0
                self.reset()
        self.net.train()

    def eval_episodes(self):
        """
        Evaluates the agent's performance over a number of episodes.
        """
        self.net.eval()
        rewards = []
        for _ in range(self.config['eval_episodes']):
            self.reset()
            while not self.env.done:
                epsilon = self.config['epsilon_by_step'](self.total_steps)
                action = self.net.act(to_var(self.state), epsilon)
                next_state, reward, done = self.env.step(action, True)
                self.total_reward += reward
                self.episode_reward += reward
                self.state = next_state
            rewards.append(self.episode_reward)
        self.logger.info(f'step {self.total_steps}, eval episodes {self.config["eval_episodes"]}, avg. reward {np.mean(rewards):.1f}')
        self.net.train()

    def save(self, path):
        """
        Saves the agent's state to a file.

        :param path: Path to save the state to.
        """
        state = {
            'total_steps': self.total_steps,
            'memory': self.memory,
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'env': self.env.state_dict(),
            'episode_steps': self.episode_steps,
            'episode_reward': self.episode_reward,
            'episode_num': self.episode_num
        }
        torch.save(state, path)

    def load(self, path):
        """
        Loads the agent's state from a file.

        :param path: Path to load the state from.
        """
        state = torch.load(path)
        self.total_steps = state['total_steps']
        self.memory = state['memory']
        self.net.load_state_dict(state['net'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.env.load_state_dict(state['env'])
        self.episode_steps = state['episode_steps']
        self.episode_reward = state['episode_reward']
        self.episode_num = state['episode_num']

    def close(self):
        """
        Closes the agent and cleans up any resources.
        """
        self.env.close()
        self.logger.handlers = []
        close_obj(self.logger)


class Transition:
    def __init__(self, *args):
        """
        Represents a transition between states in the experience replay memory.

        :param args: Tuple of (state, action, reward, next_state, done) values.
        """
        self.state, self.action, self.reward, self.next_state, self.done = zip(*args)

    def __len__(self):
        """
        Gets the number of transitions in the batch.

        :return: Number of transitions.
        """
        return len(self.done)


class QNet(torch.nn.Module):
    def __init__(self, state_shape, num_actions):
        """
        Represents a Q-value network.

        :param state_shape: Shape of the state tensor.
        :param num_actions: Number of possible actions.
        """
        super().__init__()
        self.conv1 = torch.nn.Conv2d(state_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = torch.nn.Linear(self.conv_output_size(state_shape), 512)
        self.fc2 = torch.nn.Linear(512, num_actions)
        self.apply(weights_init)

    def forward(self, x):
        """
        Computes the Q-values for a batch of states.

        :param x: Batch of states.
        :return: Batch of Q-values.
        """
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.fc2(x)
        return x
