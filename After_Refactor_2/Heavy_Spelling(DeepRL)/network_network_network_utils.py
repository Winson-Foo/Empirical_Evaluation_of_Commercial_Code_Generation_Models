import torch
import torch.optim as optim
import random
from collections import namedtuple

from models import DQN
from memory import ReplayMemory
from utils import get_state, get_action, get_reward, optimize_model
from config import Config


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def main():
    # set up environment and agent
    env = gym.make('CartPole-v1')
    env.seed(0)
    random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=Config.LEARNING_RATE)
    memory = ReplayMemory(Config.MEMORY_CAPACITY)

    # training loop
    steps_done = 0
    for i_episode in range(num_episodes):
        # initialize state
        state = get_state(env.reset())

        for t in count():
            # select action based on epsilon-greedy policy
            action = get_action(state, agent, steps_done)

            # take action and observe next state and reward
            next_state, reward, done, _ = env.step(action.item())
            next_state = get_state(next_state)
            reward = get_reward(reward, done)

            # add transition to memory
            memory.push(Transition(state, action, next_state, reward))

            # move to next state
            state = next_state

            # optimize model
            if len(memory) > Config.BATCH_SIZE:
                optimize_model(agent, memory, optimizer)

            # update exploration rate
            steps_done += 1
            if steps_done % Config.EPS_DECAY == 0:
                agent.update_epsilon(steps_done)

            # update target network
            if steps_done % Config.TARGET_UPDATE == 0:
                agent.update_target()

            # end episode if done
            if done:
                break

    env.close()


if __name__ == '__main__':
    num_episodes = 1000
    main()