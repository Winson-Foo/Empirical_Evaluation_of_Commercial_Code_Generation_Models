from examples import *
import itertools

ATARI_GAMES = [
    'BreakoutNoFrameskip-v4',
    'AlienNoFrameskip-v4',
    'DemonAttackNoFrameskip-v4',
    'MsPacmanNoFrameskip-v4'
]

MUJOCO_GAMES = [
    'HalfCheetah-v2',
    'Walker2d-v2',
    'Swimmer-v2',
    'Hopper-v2',
    'Reacher-v2',
    'Ant-v2',
    'Humanoid-v2',
    'HumanoidStandup-v2',
]

ALGOS_ATARI = [
    dqn_pixel,
    quantile_regression_dqn_pixel,
    categorical_dqn_pixel,
    rainbow_pixel,
    a2c_pixel,
    n_step_dqn_pixel,
    option_critic_pixel,
    ppo_pixel,
]

ALGOS_HUMANOID = [ppo_continuous]

ALGOS_OTHERS = [ppo_continuous, ddpg_continuous, td3_continuous]

PARAMS_ATARI = list(itertools.product(ALGOS_ATARI, ATARI_GAMES, range(1)))
PARAMS_HUMANOID = list(itertools.product(ALGOS_HUMANOID, MUJOCO_GAMES[:1], range(5)))
PARAMS_OTHERS = list(itertools.product(ALGOS_OTHERS, MUJOCO_GAMES[1:], range(5)))

PARAMS = PARAMS_ATARI + PARAMS_HUMANOID + PARAMS_OTHERS

def batch_atari():
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    algo, params = PARAMS[cf.i]
    game, run = params['game'], params['run']
    algo(game=game, run=run, remark=algo.__name__)

    exit()

def batch_mujoco():
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    algo, params = PARAMS[cf.i + len(PARAMS_ATARI)]
    game, run = params['game'], params['run']
    algo(game=game, run=run, remark=algo.__name__)

    exit()

if __name__ == '__main__':
    select_device(-1)
    batch_mujoco()