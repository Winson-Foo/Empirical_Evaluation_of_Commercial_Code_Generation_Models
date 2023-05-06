import matplotlib.pyplot as plt
from deep_rl import Plotter

def plot_game(games, patterns, labels, agg, downsample, right_align, tag, root, interpolation, window, path):
    plotter = Plotter()
    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg=agg,
                       downsample=downsample,
                       labels=labels,
                       right_align=right_align,
                       tag=tag,
                       root=root,
                       interpolation=interpolation,
                       window=window)

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')

def plot_ppo():
    games = ['HalfCheetah-v2','Walker2d-v2','Hopper-v2','Swimmer-v2','Reacher-v2','Ant-v2','Humanoid-v2','HumanoidStandup-v2',]
    patterns = ['remark_ppo']
    labels = ['PPO']
    plot_game(games=games,
              patterns=patterns,
              labels=labels,
              agg='mean',
              downsample=0,
              right_align=False,
              tag=Plotter.RETURN_TRAIN,
              root='./data/benchmark/mujoco',
              interpolation=100,
              window=10,
              path='images/PPO.png')

def plot_ddpg_td3():
    games = ['HalfCheetah-v2','Walker2d-v2','Hopper-v2','Swimmer-v2','Reacher-v2','Ant-v2']
    patterns = ['remark_ddpg', 'remark_td3']
    labels = ['DDPG', 'TD3']
    plot_game(games=games,
              patterns=patterns,
              labels=labels,
              agg='mean',
              downsample=0,
              right_align=False,
              tag=Plotter.RETURN_TEST,
              root='./data/benchmark/mujoco',
              interpolation=0,
              window=0,
              path='images/mujoco_eval.png')

def plot_atari():
    games = ['BreakoutNoFrameskip-v4']
    patterns = ['remark_a2c', 'remark_categorical', 'remark_dqn', 'remark_n_step_dqn', 'remark_option_critic', 'remark_quantile', 'remark_ppo']
    labels = ['A2C', 'C51', 'DQN', 'N-Step DQN', 'OC', 'QR-DQN', 'PPO']
    plot_game(games=games,
              patterns=patterns,
              labels=labels,
              agg='mean',
              downsample=100,
              right_align=False,
              tag=Plotter.RETURN_TRAIN,
              root='./data/benchmark/atari',
              interpolation=0,
              window=100,
              path='images/Breakout.png')

if __name__ == '__main__':
    from os import mkdir, path
    if not path.exists('images'):
        mkdir('images')
    plot_ppo()
    # plot_ddpg_td3()
    # plot_atari()