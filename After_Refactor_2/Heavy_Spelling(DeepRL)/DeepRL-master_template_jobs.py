from examples import *
import itertools


def get_atari_params(games, algos):
    params = []

    for game in games:
        for r in range(1):
            for algo in algos:
                params.append([algo, dict(game=game, run=r, remark=algo.__name__)])

    return params


def get_mujoco_params(games, algos):
    params = []

    for game in games:
        for algo in algos:
            for r in range(5):
                params.append([algo, dict(game=game, run=r)])

    return params


def run_experiments(params):
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    algo, param = params[cf.i]
    algo(**param, remark=algo.__name__)
    exit()


if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    random_seed()

    # Atari experiments
    atari_games = ['BreakoutNoFrameskip-v4']
    atari_algos = [dqn_pixel,
                   quantile_regression_dqn_pixel,
                   categorical_dqn_pixel,
                   rainbow_pixel,
                   a2c_pixel,
                   n_step_dqn_pixel,
                   option_critic_pixel,
                   ppo_pixel]

    atari_params = get_atari_params(atari_games, atari_algos)
    run_experiments(atari_params)

    # Mujoco experiments
    mujoco_games = ['HalfCheetah-v2',
                    'Walker2d-v2',
                    'Swimmer-v2',
                    'Hopper-v2',
                    'Reacher-v2',
                    'Ant-v2',
                    'Humanoid-v2',
                    'HumanoidStandup-v2']

    mujoco_algos = [ppo_continuous, ddpg_continuous, td3_continuous]

    mujoco_params = get_mujoco_params(mujoco_games, mujoco_algos)
    select_device(-1)
    run_experiments(mujoco_params)