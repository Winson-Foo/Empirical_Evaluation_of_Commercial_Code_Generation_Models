import matplotlib.pyplot as plt
from deep_rl import *

# Constants
MUJOCO_ROOT = './data/benchmark/mujoco'
ATARI_ROOT = './data/benchmark/atari'
IMAGES_ROOT = 'images'

MUJOCO_GAMES = [
    'HalfCheetah-v2',
    'Walker2d-v2',
    'Hopper-v2',
    'Swimmer-v2',
    'Reacher-v2',
    'Ant-v2',
    'Humanoid-v2',
    'HumanoidStandup-v2',
]

MUJOCO_LABELS_PPO = [
    'PPO'
]

MUJOCO_LABELS_DDPG_TD3 = [
    'DDPG',
    'TD3',
]

ATARI_GAMES = [
    'BreakoutNoFrameskip-v4',
]

ATARI_LABELS = [
    'A2C',
    'C51',
    'DQN',
    'N-Step DQN',
    'OC',
    'QR-DQN',
    'PPO',
]

# Plotting Functions
def plot_game(tag, games, patterns, labels, root, agg='mean', downsample=0, right_align=False, interpolation=0, window=0):
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
                       window=window,
                       )
    plt.tight_layout()
    plt.savefig(f'{IMAGES_ROOT}/{tag}.png', bbox_inches='tight')
    plt.close()

def plot_ppo():
    patterns = ['remark_ppo']
    plot_game(tag='PPO', games=MUJOCO_GAMES, patterns=patterns, labels=MUJOCO_LABELS_PPO, root=MUJOCO_ROOT, window=10)

def plot_ddpg_td3():
    patterns = ['remark_ddpg', 'remark_td3']
    plot_game(tag='DDPG_TD3', games=MUJOCO_GAMES[:6], patterns=patterns, labels=MUJOCO_LABELS_DDPG_TD3, root=MUJOCO_ROOT)

def plot_atari():
    patterns = [
        'remark_a2c',
        'remark_categorical',
        'remark_dqn',
        'remark_n_step_dqn',
        'remark_option_critic',
        'remark_quantile',
        'remark_ppo',
    ]
    plot_game(tag='Atari', games=ATARI_GAMES, patterns=patterns, labels=ATARI_LABELS, root=ATARI_ROOT, downsample=100, window=100)

# Main Function
def main():
    # Create the images directory if it does not exist
    if not os.path.exists(IMAGES_ROOT):
        os.makedirs(IMAGES_ROOT)

    # Plot all graphs
    plot_ppo()
    plot_ddpg_td3()
    plot_atari()

if __name__ == '__main__':
    main()