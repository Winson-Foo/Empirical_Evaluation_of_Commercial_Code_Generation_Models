import matplotlib.pyplot as plt
from deep_rl import *

def plot_benchmark(games, patterns, labels, tag, root, downsample=0, window=10, interpolation=100):
    plotter = Plotter()
    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean',
                       downsample=downsample,
                       labels=labels,
                       right_align=False,
                       tag=tag,
                       root=root,
                       interpolation=interpolation,
                       window=window,
                       )
    plt.tight_layout()
    plt.savefig('images/' + tag + '.png', bbox_inches='tight')

if __name__ == '__main__':
    mkdir('images')

    mujuco_games = ['HalfCheetah-v2', 'Walker2d-v2', 'Hopper-v2', 'Swimmer-v2',
                    'Reacher-v2', 'Ant-v2', 'Humanoid-v2', 'HumanoidStandup-v2']
    mujuco_patterns = ['remark_ppo']
    mujuco_labels = ['PPO']

    atari_games = ['BreakoutNoFrameskip-v4']
    atari_patterns = ['remark_a2c', 'remark_categorical', 'remark_dqn', 'remark_n_step_dqn',
                      'remark_option_critic', 'remark_quantile', 'remark_ppo']
    atari_labels = ['A2C', 'C51', 'DQN', 'N-Step DQN', 'OC', 'QR-DQN', 'PPO']

    plot_benchmark(games=mujuco_games,
                   patterns=mujuco_patterns,
                   labels=mujuco_labels,
                   tag=Plotter.RETURN_TRAIN,
                   root='./data/benchmark/mujoco')

    plot_benchmark(games=mujuco_games[:6],
                   patterns=['remark_ddpg', 'remark_td3'],
                   labels=['DDPG', 'TD3'],
                   tag=Plotter.RETURN_TEST,
                   root='./data/benchmark/mujoco')

    plot_benchmark(games=atari_games,
                   patterns=atari_patterns,
                   labels=atari_labels,
                   tag=Plotter.RETURN_TRAIN,
                   root='./data/benchmark/atari',
                   downsample=100,
                   window=100,
                   interpolation=0)