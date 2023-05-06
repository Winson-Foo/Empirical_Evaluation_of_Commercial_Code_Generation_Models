from examples import *
import itertools

# Modularity: Move configuration settings outside of batch functions
def get_atari_params():
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

    # Code smell 5: Replace list creation with generator expression
    return ([algo, dict(game=game, run=r, remark=algo.__name__)]
            for game, algo, r
            in itertools.product(games, algos, range(1)))

# Modularity: Extract common mujoco task parameters
def get_common_mujoco_params(game):
    if 'Humanoid' in game:
        return [ppo_continuous]
    else:
        return [ppo_continuous, ddpg_continuous, td3_continuous]

# Modularity: Move configuration settings outside of batch functions
def get_mujoco_params():
    # Code smell 1: Unused code
    # games = [
    #     'dm-acrobot-swingup',
    #     'dm-acrobot-swingup_sparse',
    #     'dm-ball_in_cup-catch',
    #     'dm-cartpole-swingup',
    #     'dm-cartpole-swingup_sparse',
    #     'dm-cartpole-balance',
    #     'dm-cartpole-balance_sparse',
    #     'dm-cheetah-run',
    #     'dm-finger-turn_hard',
    #     'dm-finger-spin',
    #     'dm-finger-turn_easy',
    #     'dm-fish-upright',
    #     'dm-fish-swim',
    #     'dm-hopper-stand',
    #     'dm-hopper-hop',
    #     'dm-humanoid-stand',
    #     'dm-humanoid-walk',
    #     'dm-humanoid-run',
    #     'dm-manipulator-bring_ball',
    #     'dm-pendulum-swingup',
    #     'dm-point_mass-easy',
    #     'dm-reacher-easy',
    #     'dm-reacher-hard',
    #     'dm-swimmer-swimmer15',
    #     'dm-swimmer-swimmer6',
    #     'dm-walker-stand',
    #     'dm-walker-walk',
    #     'dm-walker-run',
    # ]
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

    # Code smell 5: Replace list creation with generator expression
    return ([algo, dict(game=game, run=r)]
            for game, r, algo
            in itertools.product(games,
                                  range(5),
                                  get_common_mujoco_params(game)))

# Code smell 2: Lack of modularity
def execute_batch(params):
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

    # Code smell 3: Inconsistent variable names
    # select_device(0) # Commented out
    # execute_batch(get_atari_params()) # Commented out

    select_device(-1)
    execute_batch(get_mujoco_params())