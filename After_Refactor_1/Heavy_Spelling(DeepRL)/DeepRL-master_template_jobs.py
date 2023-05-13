import argparse
from examples import *
import os

def read_config():
    parser = argparse.ArgumentParser(description='Batch process reinforcement learning algorithms')
    parser.add_argument('--game', type=str, default='atari', help='The game to process (atari|mujojo)')
    parser.add_argument('--algo', type=str, default='all', help='The algorithm to process (default: all)')
    parser.add_argument('--run', type=int, default=0, help='The number of runs to process (default: 0)')
    args = parser.parse_args()
    return args

def preprocess_game(game):
    if game == 'atari':
        games = [
        'BreakoutNoFrameskip-v4',
        # 'AlienNoFrameskip-v4',
        # 'DemonAttackNoFrameskip-v4',
        # 'MsPacmanNoFrameskip-v4'
        ]
        algos = [
            dqn_pixel,
            quantile_regression_dqn_pixel,
            categorical_dqn_pixel,
            rainbow_pixel,
            a2c_pixel,
            n_step_dqn_pixel,
            option_critic_pixel,
            ppo_pixel,
        ]
        params = []
        for g in games:
            for r in range(1):
                for algo in algos:
                    params.append([algo, dict(game=g, run=r, remark=algo.__name__)])
        return params
    elif game == 'mujojo':
        games = [
            'HalfCheetah-v2',
            'Walker2d-v2',
            'Swimmer-v2',
            'Hopper-v2',
            'Reacher-v2',
            'Ant-v2',
            'Humanoid-v2',
            'HumanoidStandup-v2',
        ]
        algos = [ppo_continuous, ddpg_continuous, td3_continuous]
        params = []
        for g in games:
            for algo in algos:
                for r in range(5):
                    params.append([algo, dict(game=g, run=r)])
        return params

def select_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def batch_process(game, algo, run):
    if game == 'atari':
        params = preprocess_game('atari')
    elif game == 'mujojo':
        params = preprocess_game('mujojo')
    else:
        print('Invalid game selected!')
        return

    if algo != 'all':
        params = [p for p in params if p[0].__name__.lower() == algo.lower()]

    for p in params:
        p[1]['run'] = run
        p[1]['remark'] = f"{game}_{p[0].__name__}"
        p[0](**p[1])

if __name__ == '__main__':
    args = read_config()
    if args.game == 'atari' or args.game == 'mujojo':
        device = select_device()
        if not os.path.exists('log'):
            os.mkdir('log')
        if not os.path.exists('data'):
            os.mkdir('data')
        batch_process(args.game, args.algo, args.run)
    else:
        print(f"Invalid game selected: {args.game}")