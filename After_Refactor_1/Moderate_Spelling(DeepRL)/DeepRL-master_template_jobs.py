import argparse
from examples import (
    dqn_pixel, quantile_regression_dqn_pixel, categorical_dqn_pixel,
    rainbow_pixel, a2c_pixel, n_step_dqn_pixel, option_critic_pixel,
    ppo_pixel, ppo_continuous, ddpg_continuous, td3_continuous
)
from utils import mkdir, random_seed, select_device


def run_atari_games(algos, games):
    """Run reinforcement learning algorithms on Atari games."""
    params = []

    for game in games:
        for r in range(1):
            for algo in algos:
                params.append([algo, dict(game=game, run=r, remark=algo.__name__)])

    algo, param = params[0]
    algo(**param)


def run_mujoco_games(algos, games):
    """Run reinforcement learning algorithms on Mujoco games."""
    params = []

    for game in games:
        if 'Humanoid' in game:
            algos = [ppo_continuous]

        for algo in algos:
            for r in range(5):
                params.append([algo, dict(game=game, run=r)])

    algo, param = params[0]
    algo(**param, remark=algo.__name__)


def main():
    """Run the batch reinforcement learning experiments."""
    parser = argparse.ArgumentParser(description='Batch RL experiments')
    parser.add_argument('--device', type=int, default=-1,
                        help='Index of GPU to use, -1 for CPU')
    parser.add_argument('--atari', action='store_true', help='Run Atari games experiments')
    parser.add_argument('--mujoco', action='store_true', help='Run Mujoco games experiments')
    args = parser.parse_args()

    mkdir('log')
    mkdir('data')
    random_seed()

    select_device(args.device)

    if args.atari:
        games = ['BreakoutNoFrameskip-v4']
        algos = [
            dqn_pixel, quantile_regression_dqn_pixel, categorical_dqn_pixel,
            rainbow_pixel, a2c_pixel, n_step_dqn_pixel, option_critic_pixel,
            ppo_pixel,
        ]
        run_atari_games(algos, games)

    elif args.mujoco:
        games = [
            'HalfCheetah-v2', 'Walker2d-v2', 'Swimmer-v2', 'Hopper-v2',
            'Reacher-v2', 'Ant-v2', 'Humanoid-v2', 'HumanoidStandup-v2',
        ]
        algos = [ppo_continuous, ddpg_continuous, td3_continuous]
        run_mujoco_games(algos, games)


if __name__ == '__main__':
    main()