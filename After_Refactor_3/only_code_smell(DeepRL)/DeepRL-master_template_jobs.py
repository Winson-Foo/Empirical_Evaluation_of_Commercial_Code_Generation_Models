from utils import Config, mkdir, random_seed
from algorithms import *

def batch_atari():
    config_parser = Config()
    config_parser.add_argument('--i', type=int)
    config_parser.add_argument('--j', type=int)
    args = config_parser.parse_args()

    games = [
        'BreakoutNoFrameskip-v4',
        'AlienNoFrameskip-v4',
        'DemonAttackNoFrameskip-v4',
        'MsPacmanNoFrameskip-v4'
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

    params = [[algo, {'game': game_name, 'run': r, 'remark': algo_name}]
              for game_name in games
              for r in range(1)
              for idx, (algo_name, algo) in enumerate(zip(map(lambda x: x.__name__, algos), algos)))]

    algo, parameters = params[args.i]
    algo(**parameters)
    exit()


def batch_mujoco():
    config_parser = Config()
    config_parser.add_argument('--i', type=int)
    config_parser.add_argument('--j', type=int)
    args = config_parser.parse_args()

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

    params = [[algo, {'game': game_name, 'run': r}]
              for game_name in games
              for r in range(5)
              for idx, (algo_name, algo) in enumerate(zip(
                      ['ppo_continuous'] if 'Humanoid' in game_name else
                         ['ppo_continuous', 'ddpg_continuous', 'td3_continuous'],
                      [ppo_continuous] if 'Humanoid' in game_name else
                         [ppo_continuous, ddpg_continuous, td3_continuous]))
              ]

    algo, parameters = params[args.i]
    algo(**parameters, remark=algo.__name__)

    exit()


if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    random_seed()

    # select_device(0)
    # batch_atari()

    select_device(-1)
    batch_mujoco()