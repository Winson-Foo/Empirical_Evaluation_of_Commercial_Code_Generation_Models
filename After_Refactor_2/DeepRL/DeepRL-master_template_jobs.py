from examples import *
import itertools

def parse_args():
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()
    return cf

def batch_atari():
    cf = parse_args()

    games = [
        'BreakoutNoFrameskip-v4',
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

    params = itertools.product(algos, [{'game': game, 'run': r, 'remark': algo.__name__} for r in range(1)] for game in games)

    for algo, param in params:
        algo(**param)

    exit()

def batch_mujoco():
    cf = parse_args()

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

    params = []
    for game in games:
        if 'Humanoid' in game:
            algos = [ppo_continuous]
        else:
            algos = [ppo_continuous, ddpg_continuous, td3_continuous]
        for algo in algos:
            params.extend([(algo, {'game': game, 'run': r}) for r in range(5)])

    for algo, param in params:
        algo(**param, remark=algo.__name__)

    exit()

if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    random_seed()

    select_device(-1)
    batch_mujoco()