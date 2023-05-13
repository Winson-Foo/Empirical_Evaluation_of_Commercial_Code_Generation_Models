from examples import *
from typing import List, Tuple


def get_atari_params(games: List[str], algos: List[Callable]) -> List[Tuple[Callable, dict]]:
    """
    Returns a list of parameter combinations for Atari experiments
    """
    params = []
    for game in games:
        for r in range(1):
            for algo in algos:
                params.append([algo, dict(game=game, run=r, remark=algo.__name__)])
    return params


def get_mujoco_params(games: List[str], algos: List[Callable]) -> List[Tuple[Callable, dict]]:
    """
    Returns a list of parameter combinations for Mujoco experiments
    """
    params = []
    for game in games:
        if 'Humanoid' in game:
            algos = [ppo_continuous]
        else:
            algos = [ppo_continuous, ddpg_continuous, td3_continuous]
        for algo in algos:
            for r in range(5):
                params.append([algo, dict(game=game, run=r)])
    return params


def run_experiment(experiment_params: List[Tuple[Callable, dict]]) -> None:
    """
    Runs experiments with given parameter combinations
    """
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    algo, param = experiment_params[cf.i]
    algo(**param)
    exit()


def main():
    mkdir('log')
    mkdir('data')
    random_seed()

    # Atari experiments
    games_atari = ['BreakoutNoFrameskip-v4']
    algos_atari = [
        dqn_pixel,
        quantile_regression_dqn_pixel,
        categorical_dqn_pixel,
        rainbow_pixel,
        a2c_pixel,
        n_step_dqn_pixel,
        option_critic_pixel,
        ppo_pixel,
    ]
    atari_params = get_atari_params(games_atari, algos_atari)
    # run_experiment(atari_params)

    # Mujoco experiments
    games_mujoco = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Swimmer-v2',
        'Hopper-v2',
        'Reacher-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]
    algos_mujoco = [ppo_continuous, ddpg_continuous, td3_continuous]
    mujoco_params = get_mujoco_params(games_mujoco, algos_mujoco)
    run_experiment(mujoco_params)


if __name__ == '__main__':
    select_device(-1)
    main()
