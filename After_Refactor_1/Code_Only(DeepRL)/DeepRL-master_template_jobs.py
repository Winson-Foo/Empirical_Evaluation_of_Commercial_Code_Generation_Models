from examples import *
from typing import List, Dict, Tuple

def add_arguments(config: Config) -> Config:
    config.add_argument('--i', type=int, default=0)
    config.add_argument('--j', type=int, default=0)
    return config

def set_params(games: List[str], algos: List[object], algo_params: List[Dict[str, object]], final_params: List[Tuple[object, Dict[str, object]]]) -> None:
    for game in games:
        for r in range(5):
            for algo, algo_param in algos:
                params = dict(game=game, run=r, **algo_param)
                final_params.append([algo, params])

def batch_atari() -> None:
    cf = add_arguments(Config()).merge()
    games = [
        'BreakoutNoFrameskip-v4',
        # 'AlienNoFrameskip-v4',
        # 'DemonAttackNoFrameskip-v4',
        # 'MsPacmanNoFrameskip-v4'
    ]

    algos = [
        (dqn_pixel, dict(remark=dqn_pixel.__name__)),
        (quantile_regression_dqn_pixel, dict()),
        (categorical_dqn_pixel, dict(remark=categorical_dqn_pixel.__name__)),
        (rainbow_pixel, dict()),
        (a2c_pixel, dict()),
        (n_step_dqn_pixel, dict(replay_cls=PrioritizedReplay)),
        (option_critic_pixel, dict()),
        (ppo_pixel, dict())
    ]

    final_params = []
    set_params(games, algos, [], final_params)
    # set_params(games, [(dqn_pixel, dict(n_step=n_step, replay_cls=PrioritizedReplay, double_q=double_q, remark=dqn_pixel.__name__)) for n_step in [1, 2, 3] for double_q in [True, False]], final_params)
    # set_params(games, [(rainbow_pixel, dict(n_step=n_step, noisy_linear=False, remark=rainbow_pixel.__name__)) for n_step in [1, 2, 3]], final_params)

    for algo, param in final_params:
        algo(**param)
    exit()

def batch_mujoco() -> None:
    cf = add_arguments(Config()).merge()
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

    algos = [(ppo_continuous, dict()), (ddpg_continuous, dict()), (td3_continuous, dict())]
    final_params = []
    set_params(games, algos, [], final_params)
    set_params(['Humanoid-v2', 'HumanoidStandup-v2'], [(ppo_continuous, dict())], [{'lambda': 0.95}], final_params)

    for algo, param in final_params:
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
