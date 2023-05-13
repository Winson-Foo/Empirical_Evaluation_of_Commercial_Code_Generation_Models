import matplotlib.pyplot as plt
from deep_rl import Plotter

def plot_games(games, patterns, labels, tag, root, agg='mean', downsample=0, right_align=False, interpolation=0, window=10):
    plotter = Plotter()
    plotter.plot_games(
        games=games,
        patterns=patterns,
        agg=agg,
        downsample=downsample,
        labels=labels,
        right_align=right_align,
        tag=tag,
        root=root,
        interpolation=interpolation,
        window=window,
    )
    plt.tight_layout()
    plt.savefig(f'images/{tag}.png', bbox_inches='tight')

def plot_mujoco():
    games = ['HalfCheetah-v2', 'Walker2d-v2', 'Hopper-v2', 'Swimmer-v2', 'Reacher-v2', 'Ant-v2', 'Humanoid-v2', 'HumanoidStandup-v2']
    patterns = ['remark_ppo']
    labels = ['PPO']
    plot_games(games, patterns, labels, tag='PPO', root='./data/benchmark/mujoco', window=10)

    patterns = ['remark_ddpg', 'remark_td3']
    labels = ['DDPG', 'TD3']
    plot_games(games[:len(patterns)], patterns, labels, tag='mujoco_eval', root='./data/benchmark/mujoco', window=0)

def plot_atari():
    games = ['BreakoutNoFrameskip-v4']
    patterns = ['remark_a2c', 'remark_categorical', 'remark_dqn', 'remark_n_step_dqn', 'remark_option_critic', 'remark_quantile', 'remark_ppo']
    labels = ['A2C', 'C51', 'DQN', 'N-Step DQN', 'OC', 'QR-DQN', 'PPO']
    plot_games(games, patterns, labels, tag='Breakout', root='./data/benchmark/atari', downsample=100, interpolation=0, window=100)

if __name__ == '__main__':
    plt.rc('text', usetex=True)  # set rc parameters
    plt.rc('font', family='serif')
    plt.rc('font', size=14)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=14)
    plt.rc('legend', fontsize=12)
    
    plt.switch_backend('agg')  # use non-GUI backend
    plt.style.use('ggplot')  # set plot style
    plt.figure(figsize=(8, 6))  # set figure size
    plot_mujoco()
    plot_atari()