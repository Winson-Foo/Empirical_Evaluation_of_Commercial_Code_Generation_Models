import matplotlib.pyplot as plt
from deep_rl import Plotter


def plot_games(plotter, games, patterns, labels, metric, return_type, root, **kwargs):
    plotter.plot_games(games=games,
                       patterns=patterns,
                       labels=labels,
                       tag=return_type,
                       root=root,
                       metric=metric,
                       **kwargs)
    plt.tight_layout()
    filename = f"images/{return_type}_{metric}.png"
    plt.savefig(filename, bbox_inches='tight')


def plot_mujoco():
    plotter = Plotter()
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
        'remark_ddpg',
        'remark_td3',
    ]

    labels = [
        'PPO',
        'DDPG',
        'TD3',
    ]

    plot_games(plotter=plotter,
               games=games,
               patterns=patterns,
               labels=labels,
               metric='episode_reward_mean',
               return_type='train',
               root='./data/benchmark/mujoco',
               agg='mean', downsample=0, interpolation=100, window=10)

    plot_games(plotter=plotter,
               games=games[:6],
               patterns=patterns[1:],
               labels=labels[1:],
               metric='episode_return',
               return_type='test',
               root='./data/benchmark/mujoco',
               agg='mean', downsample=0, interpolation=0, window=0)


def plot_atari():
    plotter = Plotter()
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

    plot_games(plotter=plotter,
               games=games,
               patterns=patterns,
               labels=labels,
               metric='episode_reward_mean',
               return_type='train',
               root='./data/benchmark/atari',
               agg='mean', downsample=100, interpolation=0, window=100)


if __name__ == '__main__':
    plt.rc('text', usetex=True)  # Move this line from the top of the file if needed
    plt.switch_backend('agg')  # Use the non-interactive backend
    plt.style.use('ggplot')  # Apply a custom style if preferred
    plot_mujoco()
    plot_atari()