import examples

NUM_ATARI_RUNS = 1
NUM_MUJOCO_RUNS = 5
LOG_DIR = 'log'
DATA_DIR = 'data'

def create_params(game, algos):
    params = []
    for algo in algos:
        for r in range(NUM_MUJOCO_RUNS if 'Humanoid' in game else NUM_ATARI_RUNS):
            params.append([algo, {'game': game, 'run': r, 'remark': algo.__name__}])
    return params

def run_atari(games, algos):
    for game in games:
        params = create_params(game, algos)
        for algo, param in params:
            algo(**param)
    
def run_mujoco(games, algos):
    for game in games:
        params = create_params(game, algos)
        for algo, param in params:
            algo(**param)
            
if __name__ == '__main__':
    examples.mkdir(LOG_DIR)
    examples.mkdir(DATA_DIR)
    examples.random_seed()

    cf = examples.Config(i=0, j=0)
    
    examples.select_device(-1)
    
    atari_games = [
        'BreakoutNoFrameskip-v4',
        # 'AlienNoFrameskip-v4',
        # 'DemonAttackNoFrameskip-v4',
        # 'MsPacmanNoFrameskip-v4'
    ]
    
    atari_algos = [
        examples.dqn_pixel,
        examples.quantile_regression_dqn_pixel,
        examples.categorical_dqn_pixel,
        examples.rainbow_pixel,
        examples.a2c_pixel,
        examples.n_step_dqn_pixel,
        examples.option_critic_pixel,
        examples.ppo_pixel,
    ]
    
    mujoco_games = [
        'dm-acrobot-swingup', 'dm-acrobot-swingup_sparse', 'dm-ball_in_cup-catch', 'dm-cartpole-swingup',
        'dm-cartpole-swingup_sparse', 'dm-cartpole-balance', 'dm-cartpole-balance_sparse', 'dm-cheetah-run',
        'dm-finger-turn_hard', 'dm-finger-spin', 'dm-finger-turn_easy', 'dm-fish-upright', 'dm-fish-swim',
        'dm-hopper-stand', 'dm-hopper-hop', 'dm-humanoid-stand', 'dm-humanoid-walk', 'dm-humanoid-run',
        'dm-manipulator-bring_ball', 'dm-pendulum-swingup', 'dm-point_mass-easy', 'dm-reacher-easy',
        'dm-reacher-hard', 'dm-swimmer-swimmer15', 'dm-swimmer-swimmer6', 'dm-walker-stand', 'dm-walker-walk',
        'dm-walker-run',
        'HalfCheetah-v2', 'Walker2d-v2', 'Swimmer-v2', 'Hopper-v2', 'Reacher-v2', 'Ant-v2', 'Humanoid-v2',
        'HumanoidStandup-v2',
    ]
    
    mujoco_algos = [
        examples.ppo_continuous,
        examples.ddpg_continuous,
        examples.td3_continuous,
    ]
    
    run_atari(atari_games, atari_algos)
    run_mujoco(mujoco_games, mujoco_algos)