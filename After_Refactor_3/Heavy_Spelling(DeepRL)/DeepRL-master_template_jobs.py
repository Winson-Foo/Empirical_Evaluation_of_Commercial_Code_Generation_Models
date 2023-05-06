from examples import *
import itertools

GAMES_ATARI = ['BreakoutNoFrameskip-v4']
ALGOS_ATARI = [dqn_pixel, quantile_regression_dqn_pixel, categorical_dqn_pixel, rainbow_pixel, a2c_pixel, n_step_dqn_pixel, option_critic_pixel, ppo_pixel]

GAMES_MUJOCO = [
    'HalfCheetah-v2',
    'Walker2d-v2',
    'Swimmer-v2',
    'Hopper-v2',
    'Reacher-v2',
    'Ant-v2',
    'Humanoid-v2',
    'HumanoidStandup-v2',
]

ALGOS_MUJOCO = [ppo_continuous, ddpg_continuous, td3_continuous]

def create_params_game_algo(game, algo, runs=1):
    params = []
    for r in range(runs):
        params.append([algo, dict(game=game, run=r)])
    return params

def batch_atari():
    config = Config()
    config.add_argument('--i', type=int, default=0)
    config.add_argument('--j', type=int, default=0)
    config.merge()

    params = []
    for game, algo in itertools.product(GAMES_ATARI, ALGOS_ATARI):
        params.extend(create_params_game_algo(game, algo))

    algo, param = params[config.i]
    algo(**param)
    exit()

def batch_mujoco():
    config = Config()
    config.add_argument('--i', type=int, default=0)
    config.add_argument('--j', type=int, default=0)
    config.merge()

    params = []
    for game, algo in itertools.product(GAMES_MUJOCO, ALGOS_MUJOCO):
        runs = 5 if 'Humanoid' not in game else 1
        params.extend(create_params_game_algo(game, algo, runs))

    algo, param = params[config.i]
    algo(**param, remark=algo.__name__)
    exit()

if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    random_seed()

    # select_device(0)
    # batch_atari()

    select_device(-1)
    batch_mujoco()