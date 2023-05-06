from examples import *
import os

# Define constants
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
    'HumanoidStandup-v2'
]

# Define batch_atari function
def batch_atari():
    config = Config()
    config.add_argument('--i', type=int, default=0)
    config.add_argument('--j', type=int, default=0)
    config.merge()

    algo_params = [
        (dqn_pixel, dict(remark=dqn_pixel.__name__)),
        (quantile_regression_dqn_pixel, dict(remark=quantile_regression_dqn_pixel.__name__)),
        (categorical_dqn_pixel, dict(remark=categorical_dqn_pixel.__name__)),
        (rainbow_pixel, dict(noisy_linear=False, remark=rainbow_pixel.__name__)),
        (a2c_pixel, dict()),
        (n_step_dqn_pixel, dict(replay_cls=PrioritizedReplay)),
        (option_critic_pixel, dict()),
        (ppo_pixel, dict())
    ]

    for i, game in enumerate(ATARI_GAMES):
        for r in range(1):
            for algo, algo_kwargs in algo_params:
                params = dict(game=game, run=r, **algo_kwargs)
                algo(**params)

# Define batch_mujoco function
def batch_mujoco():
    config = Config()
    config.add_argument('--i', type=int, default=0)
    config.add_argument('--j', type=int, default=0)
    config.merge()

    algo_params = [
        (ppo_continuous, {}),
        (ddpg_continuous, {}),
        (td3_continuous, {})
    ]

    for i, game in enumerate(MUJOCO_GAMES):
        if 'Humanoid' in game:
            algos = [ppo_continuous]
        else:
            algos = [ppo_continuous, ddpg_continuous, td3_continuous]
        for algo, _ in algos:
            for r in range(5):
                params = dict(game=game, run=r)
                algo(**params, remark=algo.__name__)

# Execute batch_atari or batch_mujoco function
def run_batch():
    random_seed()
    mkdir('log')
    mkdir('data')

    device = int(input("Please enter a device (-1 for CPU, 0 for GPU): "))
    select_device(device)

    batch_type = input("Please enter batch type (atari or mujoco): ")
    if batch_type.lower() == "atari":
        batch_atari()
    elif batch_type.lower() == "mujoco":
        batch_mujoco()
    else:
        print("Invalid batch type. Please enter either 'atari' or 'mujoco'.")

if __name__ == '__main__':
    run_batch()