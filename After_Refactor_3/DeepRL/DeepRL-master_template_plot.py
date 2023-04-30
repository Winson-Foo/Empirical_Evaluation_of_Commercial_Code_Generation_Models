import matplotlib.pyplot as plt
from deep_rl import Plotter

def plot_games(games, patterns, labels, tag, root):
    plotter = Plotter()
    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean',
                       downsample=0,
                       labels=labels,
                       right_align=False,
                       tag=tag,
                       root=root,
                       interpolation=100,
                       window=10,
                       )
    plt.tight_layout()

def save_figure(filename):
    plt.savefig(f'images/{filename}.png', bbox_inches='tight')

def plot_ppo():
    game_patterns = {
        'HalfCheetah-v2': ['remark_ppo'],
        'Walker2d-v2': ['remark_ppo'],
        'Hopper-v2': ['remark_ppo'],
        'Swimmer-v2': ['remark_ppo'],
        'Reacher-v2': ['remark_ppo'],
        'Ant-v2': ['remark_ppo'],
        'Humanoid-v2': ['remark_ppo'],
        'HumanoidStandup-v2': ['remark_ppo']
    }

    game_labels = {'PPO'}

    plot_games(games=game_patterns.keys(),
               patterns=list(game_patterns.values()),
               labels=game_labels,
               tag=Plotter.RETURN_TRAIN,
               root='./data/benchmark/mujoco')
    save_figure('PPO')

def plot_ddpg_td3():
    game_patterns = {
        'HalfCheetah-v2': ['remark_ddpg', 'remark_td3'],
        'Walker2d-v2': ['remark_ddpg', 'remark_td3'],
        'Hopper-v2': ['remark_ddpg', 'remark_td3'],
        'Swimmer-v2': ['remark_ddpg', 'remark_td3'],
        'Reacher-v2': ['remark_ddpg', 'remark_td3'],
        'Ant-v2': ['remark_ddpg', 'remark_td3']
    }

    game_labels = {'DDPG', 'TD3'}

    plot_games(games=game_patterns.keys(),
               patterns=list(game_patterns.values()),
               labels=game_labels,
               tag=Plotter.RETURN_TEST,
               root='./data/benchmark/mujoco')
    save_figure('mujoco_eval')

def plot_atari():
    game_patterns = {
        'BreakoutNoFrameskip-v4': ['remark_a2c', 'remark_categorical', 'remark_dqn', 'remark_n_step_dqn', 'remark_option_critic',
                                   'remark_quantile', 'remark_ppo']
    }

    game_labels = {'A2C', 'C51', 'DQN', 'N-Step DQN', 'OC', 'QR-DQN', 'PPO'}
    plot_games(games=game_patterns.keys(),
               patterns=list(game_patterns.values()),
               labels=game_labels,
               tag=Plotter.RETURN_TRAIN,
               root='./data/benchmark/atari')
    save_figure('Breakout')

if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=10)
    plt.rc('axes', titlesize=10)
    plt.rc('axes', labelsize=10)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('legend', fontsize=10)
    plt.rc('figure', titlesize=12)
    
    plot_ppo()
    plot_ddpg_td3()
    plot_atari()