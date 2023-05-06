# main.py
import torch.multiprocessing as mp
import gym

from component import Replay, Transition
from network import Network, CategoricalNetwork
from utils import to_numpy, to_tensor
from agent import DQNAgent, CategoricalDQNAgent

def train(env_name, agent_class, config):
    env = gym.make(env_name)

    config.state_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.n
    config.delta_z = (config.categorical_v_max - config.categorical_v_min) / (config.categorical_n_atoms - 1)

    agent = agent_class(config)

    state = env.reset()
    episode_reward = 0
    episode_steps = 0
    episode_num = 0

    for step in range(config.total_steps):
        epsilon = max(config.epsilon_final, config.epsilon_initial - step / config.epsilon_decay_steps)

        action = agent.eval_step(state)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        episode_steps += 1

        agent.learn(state, action, reward, next_state, done)

        if done:
            episode_num += 1
            print("Episode {}: reward={}, steps={}".format(episode_num, episode_reward, episode_steps))
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
        else:
            state = next_state

    env.close()

def main():
    config = {
        'env_name': 'CartPole-v0',
        'agent_class': 'DQNAgent',
        'replay_capacity': 10000,
        'replay_start_size': 1000,
        'batch_size': 32,
        'learning_rate': 0.001,
        'discount': 0.99,
        'target_update_freq': 1000,
        'total_steps': 100000,
        'epsilon_initial': 1.0,
        'epsilon_final': 0.01,
        'epsilon_decay_steps': 10000,
        'categorical_v_min': -10,
        'categorical_v_max': 10,
        'categorical_n_atoms': 51,
    }
    config = Config(**config)

    if config.agent_class == 'DQNAgent':
        agent_class = DQNAgent
    elif config.agent_class == 'CategoricalDQNAgent':
        agent_class = CategoricalDQNAgent
    else:
        raise ValueError("Unknown agent class: {}".format(config.agent_class))

    train(config.env_name, agent_class, config)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()