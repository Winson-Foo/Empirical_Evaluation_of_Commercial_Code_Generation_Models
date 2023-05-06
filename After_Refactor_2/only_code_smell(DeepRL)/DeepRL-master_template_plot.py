import matplotlib.pyplot as plt
from deep_rl import *

def plot_game_results(games, patterns, labels, return_type, root, output_file):
    plotter = Plotter()
    
    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean',
                       downsample=0 if return_type == 'test' else 100,
                       labels=labels,
                       right_align=False,
                       tag=[plotter.RETURN_TRAIN, plotter.RETURN_TEST][return_type=='test'],
                       root=root,
                       interpolation=[100,0][return_type=='test'],
                       window=10 if return_type=='train' else 100,
                       )
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')


def plot_ppo():
    games = ['HalfCheetah-v2', 'Walker2d-v2', 'Hopper-v2', 'Swimmer-v2', 'Reacher-v2', 'Ant-v2', 'Humanoid-v2', 'HumanoidStandup-v2']
    patterns = ['remark_ppo']
    labels = ['PPO']
    output_file = 'images/PPO.png'
    root = './data/benchmark/mujoco'
    
    plot_game_results(games, patterns, labels, 'train', root, output_file)

def plot_ddpg_td3():
    games = ['HalfCheetah-v2', 'Walker2d-v2', 'Hopper-v2', 'Swimmer-v2', 'Reacher-v2', 'Ant-v2']
    patterns = ['remark_ddpg', 'remark_td3']
    labels = ['DDPG', 'TD3']
    output_file = 'images/mujoco_eval.png'
    root = './data/benchmark/mujoco'
    
    plot_game_results(games, patterns, labels, 'test', root, output_file)
    

def plot_atari():
    games = ['BreakoutNoFrameskip-v4']
    patterns = ['remark_a2c', 'remark_categorical', 'remark_dqn', 'remark_n_step_dqn', 'remark_option_critic', 'remark_quantile', 'remark_ppo']
    labels = ['A2C', 'C51', 'DQN', 'N-Step DQN', 'OC', 'QR-DQN', 'PPO']
    output_file = 'images/Breakout.png'
    root = './data/benchmark/atari'
    
    plot_game_results(games, patterns, labels, 'train', root, output_file)

if __name__ == '__main__':
    mkdir('images')
    plot_ppo()
    plot_ddpg_td3()
    plot_atari()
