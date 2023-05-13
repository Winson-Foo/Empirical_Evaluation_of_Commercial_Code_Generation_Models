import matplotlib.pyplot as plt
from deep_rl import *


def plot_game(games, patterns, labels, tag, root, downsample, interpolation, window)
    """
    Plot the game graphs. 
    :param games: list of games
    :param patterns: list of patterns to be used
    :param labels: list of corresponding labels to be used
    :param tag: type of game to plot (train or test)
    :param root: root directory of data
    :param downsample: frequency at which to downsample data
    :param interpolation: type of spline interpolation to be used
    :param window: window size for moving average
    :return: None
    """
    plotter = Plotter()
    plotter.plot_games(
        games=games,
        patterns=patterns,
        agg='mean',
        downsample=downsample,
        labels=labels,
        right_align=False,
        tag=tag,
        root=root,
        interpolation=interpolation,
        window=window
    )

    plt.tight_layout()
    plt.savefig(f'images/{games[0]}_{tag}.png', bbox_inches='tight')


def plot_mujoco():
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Swimmer-v2',
        'Reacher-v2',
        'Ant-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
    ]

    patterns = [
        'remark_ppo',
    ]

    labels = [
        'PPO'
    ]

    plot_game(
        games=games,
        patterns=patterns,
        labels=labels,
        tag=Plotter.RETURN_TRAIN,
        root='./data/benchmark/mujoco',
        downsample=0,
        interpolation=100,
        window=10
    )


def plot_mujoco_agents():
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Swimmer-v2',
        'Reacher-v2',
        'Ant-v2',
    ]

    patterns = [
        'remark_ddpg',
        'remark_td3',
    ]

    labels = [
        'DDPG',
        'TD3',
    ]

    plot_game(
        games=games,
        patterns=patterns,
        labels=labels,
        tag=Plotter.RETURN_TEST,
        root='./data/benchmark/mujoco',
        downsample=0,
        interpolation=0,
        window=0
    )


def plot_atari():
    games = [
        'BreakoutNoFrameskip-v4',
    ]

    patterns = [
        'remark_a2c',
        'remark_categorical',
        'remark_dqn',
        'remark_n_step_dqn',
        'remark_option_critic',
        'remark_quantile',
        'remark_ppo',
        # 'remark_rainbow',
    ]

    labels = [
        'A2C',
        'C51',
        'DQN',
        'N-Step DQN',
        'OC',
        'QR-DQN',
        'PPO',
        # 'Rainbow'
    ]

    plot_game(
        games=games,
        patterns=patterns,
        labels=labels,
        tag=Plotter.RETURN_TRAIN,
        root='./data/benchmark/atari',
        downsample=100,
        interpolation=0,
        window=100
    )


if __name__ == '__main__':
    mkdir('images')
    plot_mujoco()
    plot_mujoco_agents()
    plot_atari()