# file: main.py
import argparse
import logging

from task import Task
from utils import mkdir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Hopper-v2', help='name of the gym environment')
    parser.add_argument('--num_envs', type=int, default=1, help='number of environments to run in parallel')
    parser.add_argument('--single_process', action='store_true', help='run environment in a single process')
    parser.add_argument('--log_dir', type=str, default=None, help='directory to save log files')
    parser.add_argument('--episode_life', action='store_true', help='use episode life mechanism in atari environment')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()

    mkdir(args.log_dir)

    logging.basicConfig(level=logging.INFO)

    task = Task(args.env_name, args.num_envs, args.single_process, args.log_dir, args.episode_life, args.seed)

    state = task.reset()
    while True:
        action = np.random.rand(task.observation_space.shape[0])
        next_state, reward, done, _ = task.step(action)
        logging.debug(done)

if __name__ == '__main__':
    main()