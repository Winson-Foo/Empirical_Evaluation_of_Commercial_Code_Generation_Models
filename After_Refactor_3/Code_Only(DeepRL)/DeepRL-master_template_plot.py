import matplotlib.pyplot as plt
from deep_rl import Plotter


def plot_game(plotter, games, patterns, labels, tag, root, agg='mean', downsample=0, interpolation=0, window=0, right_align=False):
    plotter.plot_games(games=games, patterns=patterns, agg=agg, downsample=downsample,
                       labels=labels, right_align=right_align, tag=tag, root=root,
                       interpolation=interpolation, window=window)
    plt.tight_layout()
    plt.savefig(f'images/{tag}.png', bbox_inches='tight')


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
    root = './data/benchmark/mujoco'
    plotter = Plotter()
    plot_game(plotter, games, patterns, labels, plotter.RETURN_TRAIN, root, agg='mean', downsample=0, interpolation=100, window=10)


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
    root = './data/benchmark/mujoco'
    plotter = Plotter()
    plot_game(plotter, games, patterns, labels, plotter.RETURN_TEST, root, agg='mean', downsample=0, interpolation=0, window=0)

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
    root = './data/benchmark/atari'
    plotter = Plotter()
    plot_game(plotter, games, patterns, labels, plotter.RETURN_TRAIN, root, agg='mean', downsample=100, interpolation=0, window=100)


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=15)
    plt.rc('axes', titlesize=15)
    plt.rc('axes', labelsize=15)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('legend', fontsize=15)
    plt.rc('figure', titlesize=15)
    plt.rc('figure', figsize=(6, 4))
    plt.rc('savefig', dpi=300)
    plt.rc('savefig', bbox='tight')
    plt.rc('savefig', format='png')
    plt.rc('savefig', pad_inches=0.1)
    plt.rc('savefig', facecolor='white')
    plot_ppo()
    plot_ddpg_td3()
    plot_atari()