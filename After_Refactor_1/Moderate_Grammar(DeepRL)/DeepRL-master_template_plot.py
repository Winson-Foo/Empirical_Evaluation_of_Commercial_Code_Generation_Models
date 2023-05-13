import matplotlib.pyplot as plt
from deep_rl import Plotter

# Constants
MUJOCO_ROOT_DIR = './data/benchmark/mujoco'
ATARI_ROOT_DIR = './data/benchmark/atari'
IMAGES_DIR = 'images'
GAMES_MUJOCO = [
    'HalfCheetah-v2',
    'Walker2d-v2',
    'Hopper-v2',
    'Swimmer-v2',
    'Reacher-v2',
    'Ant-v2',
    'Humanoid-v2',
    'HumanoidStandup-v2',
]
GAMES_ATARI = [
    'BreakoutNoFrameskip-v4',
]
PATTERNS_PPO = [
    'remark_ppo',
]
PATTERNS_DDPG_TD3 = [
    'remark_ddpg',
    'remark_td3',
]
PATTERNS_ATARI = [
    'remark_a2c',
    'remark_categorical',
    'remark_dqn',
    'remark_n_step_dqn',
    'remark_option_critic',
    'remark_quantile',
    'remark_ppo',
]
LABELS_PPO = [
    'PPO'
]
LABELS_DDPG_TD3 = [
    'DDPG',
    'TD3',
]
LABELS_ATARI = [
    'A2C',
    'C51',
    'DQN',
    'N-Step DQN',
    'OC',
    'QR-DQN',
    'PPO',
]

# Functions
def plot_game(games, patterns, labels, return_type, root_dir, downsample=0, interpolation=0, window=0):
    plotter = Plotter()
    plotter.plot_games(games=games,
                   patterns=patterns,
                   agg='mean',
                   downsample=downsample,
                   labels=labels,
                   right_align=False,
                   tag=return_type,
                   root=root_dir,
                   interpolation=interpolation,
                   window=window,
                   )
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/{games[0]}_{return_type}.png', bbox_inches='tight')

def plot_mujoco():
    plot_game(games=GAMES_MUJOCO, 
              patterns=PATTERNS_PPO, 
              labels=LABELS_PPO, 
              return_type=Plotter.RETURN_TRAIN, 
              root_dir=MUJOCO_ROOT_DIR)
    # plot_game(games=GAMES_MUJOCO, 
    #          patterns=PATTERNS_DDPG_TD3, 
    #          labels=LABELS_DDPG_TD3, 
    #          return_type=Plotter.RETURN_TEST, 
    #          root_dir=MUJOCO_ROOT_DIR)

def plot_atari():
    plot_game(games=GAMES_ATARI, 
              patterns=PATTERNS_ATARI, 
              labels=LABELS_ATARI, 
              return_type=Plotter.RETURN_TRAIN, 
              root_dir=ATARI_ROOT_DIR, 
              downsample=100,
              window=100)

if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'figure.autolayout': True})
    plt.rcParams.update({'figure.figsize': (5, 5)})
    plt.rcParams.update({'lines.linewidth': 2.0})
    
    # Create images directory
    mkdir(IMAGES_DIR)

    # Plot games
    plot_mujoco()
    plot_atari()