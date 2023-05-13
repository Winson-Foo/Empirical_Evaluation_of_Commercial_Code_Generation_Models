from examples import *
from typing import Dict, Any, Tuple, List

ATARI_GAMES = [
    'BreakoutNoFrameskip-v4',
    # 'AlienNoFrameskip-v4',
    # 'DemonAttackNoFrameskip-v4',
    # 'MsPacmanNoFrameskip-v4'
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

ALGOS = {
    'atari': [
        dqn_pixel,
        quantile_regression_dqn_pixel,
        categorical_dqn_pixel,
        rainbow_pixel,
        a2c_pixel,
        n_step_dqn_pixel,
        option_critic_pixel,
        ppo_pixel,
    ],
    'mujoco': [ppo_continuous, ddpg_continuous, td3_continuous]
}

def run_algos(game_params: Dict[str, Any], env: str, algos: List,
              num_runs: int = 1) -> None:
    params = []
    for game_name, game_remark in game_params.items():
        for r in range(num_runs):
            for algo in algos:
                params.append([algo, dict(game=game_name, run=r, remark=game_remark + algo.__name__)])
    algo, param = params[0]
    algo(**param)

def batch_atari() -> None:
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    game_params = {game: f'atari_{game}_' for game in ATARI_GAMES}

    run_algos(game_params, 'atari', ALGOS['atari'])

    exit()

def batch_mujoco() -> None:
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    game_params = {game: f'mujoco_{game}_' for game in MUJOCO_GAMES}

    run_algos(game_params, 'mujoco', ALGOS['mujoco'], num_runs=5)

    exit()

if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    random_seed()

    # select_device(0)
    # batch_atari()

    select_device(-1)
    batch_mujoco()