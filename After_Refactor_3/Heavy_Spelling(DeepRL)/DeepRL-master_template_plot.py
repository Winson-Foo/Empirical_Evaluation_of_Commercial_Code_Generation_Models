import matplotlib.pyplot as plt
from deep_rl import Plotter

ROOT = './data/benchmark'
AGG = 'mean'
TAG = Plotter.RETURN_TRAIN


def plot_games(games, patterns, labels, downsample=0, interpolation=0, window=0, right_align=False):
    plotter = Plotter()
    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg=AGG,
                       downsample=downsample,
                       labels=labels,
                       right_align=right_align,
                       tag=TAG,
                       root=ROOT,
                       interpolation=interpolation,
                       window=window,
                       )

    with plt.rc_context({'figure.figsize': (10, 5)}):
        plt.tight_layout()
        plt.savefig('images/plot.png', bbox_inches='tight')


def plot_ppo():
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

    plot_games(games=games, patterns=patterns, labels=labels, right_align=True)


def plot_ddpg_td3():
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

    plot_games(games=games, patterns=patterns, labels=labels)


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
    ]

    labels = [
        'A2C',
        'C51',
        'DQN',
        'N-Step DQN',
        'OC',
        'QR-DQN',
        'PPO',
    ]

    plot_games(games=games, patterns=patterns, labels=labels, downsample=100, window=100)


if __name__ == '__main__':
    plt.switch_backend('agg')
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    plot_ppo()
    plot_ddpg_td3()
    plot_atari()