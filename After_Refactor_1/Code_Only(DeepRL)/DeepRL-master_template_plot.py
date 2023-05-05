import matplotlib.pyplot as plt
from deep_rl import Plotter

def plot_graphs(games, patterns, labels, tag, root, agg='mean', downsample=0, 
                right_align=False, interpolation=0, window=0, filename):
    plotter = Plotter()
    plotter.plot_games(games=games, patterns=patterns, agg=agg, downsample=downsample, 
                       labels=labels, right_align=right_align, tag=tag, root=root, 
                       interpolation=interpolation, window=window)

    plt.tight_layout()
    plt.savefig(f'images/{filename}.png', bbox_inches='tight')

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

    plot_graphs(games, patterns, labels, Plotter.RETURN_TRAIN, 
                './data/benchmark/mujoco', filename='PPO')

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

    plot_graphs(games, patterns, labels, Plotter.RETURN_TEST, './data/benchmark/mujoco', 
                filename='DDPG_TD3')

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

    plot_graphs(games, patterns, labels, Plotter.RETURN_TRAIN, './data/benchmark/atari', 
                downsample=100, interpolation=0, window=100, filename='Breakout')

if __name__ == '__main__':
    if not os.path.exists('images'):
        os.makedirs('images')
    plot_ppo()
    plot_ddpg_td3()
    plot_atari()