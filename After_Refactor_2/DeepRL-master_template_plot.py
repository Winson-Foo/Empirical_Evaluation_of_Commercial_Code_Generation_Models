import matplotlib.pyplot as plt
from deep_rl import *


class Plotting:
    """Encapsulates common variables and methods for plotting functions"""

    RETURN_TRAIN = 'return_train'
    RETURN_TEST = 'return_test'
    GAMES = {
        'Mujoco': [
            'HalfCheetah-v2',
            'Walker2d-v2',
            'Hopper-v2',
            'Swimmer-v2',
            'Reacher-v2',
            'Ant-v2',
            'Humanoid-v2',
            'HumanoidStandup-v2',
        ],
        'Atari': [
            'BreakoutNoFrameskip-v4',
        ]
    }
    PATTERNS = {
        'Mujoco': [
            'remark_ppo',
            'remark_ddpg',
            'remark_td3',
        ],
        'Atari': [
            'remark_a2c',
            'remark_categorical',
            'remark_dqn',
            'remark_n_step_dqn',
            'remark_option_critic',
            'remark_quantile',
            'remark_ppo',
            # 'remark_rainbow',
        ]
    }
    LABELS = {
        'Mujoco': [
            'PPO',
            'DDPG',
            'TD3',
        ],
        'Atari': [
            'A2C',
            'C51',
            'DQN',
            'N-Step DQN',
            'OC',
            'QR-DQN',
            'PPO',
            # 'Rainbow'
        ]
    }

    def __init__(self, game, pattern, label):
        self._game = Plotting.GAMES[game]
        self._pattern = Plotting.PATTERNS[game][pattern]
        self._label = Plotting.LABELS[game][label]

    def plot_game(self, tag, root, y_label, save_file):
        """Plot a single game"""
        plotter = Plotter()
        plotter.plot_games(games=self._game,
                           patterns=[self._pattern],
                           agg='mean',
                           downsample=0,
                           labels=[self._label],
                           right_align=False,
                           tag=tag,
                           root=root,
                           interpolation=100,
                           window=10,
                           )
        
        plt.xlabel('Steps')
        plt.ylabel(y_label)
        plt.tight_layout()
        plt.savefig(save_file, bbox_inches='tight')

    def plot_all_games(self, tag, root, y_label, save_file):
        """Plot all games for a given pattern and label"""
        plotter = Plotter()
        plotter.plot_games(games=self._game,
                           patterns=[self._pattern],
                           agg='mean',
                           downsample=100,
                           labels=[self._label],
                           right_align=False,
                           tag=tag,
                           root=root,
                           interpolation=0,
                           window=100,
                           )
        
        plt.xlabel('Steps')
        plt.ylabel(y_label)
        plt.tight_layout()
        plt.savefig(save_file, bbox_inches='tight')


def plot_mujoco():
    Plotting('Mujoco', 0, 0).plot_all_games(
        tag=Plotting.RETURN_TRAIN,
        root='./data/benchmark/mujoco',
        y_label='Average return',
        save_file='images/PPO.png'
    )

    Plotting('Mujoco', 1, 1).plot_all_games(
        tag=Plotting.RETURN_TEST,
        root='./data/benchmark/mujoco',
        y_label='Average return in test',
        save_file='images/mujoco_eval.png'
    )


def plot_atari():
    Plotting('Atari', 0, 0).plot_game(
        tag=Plotting.RETURN_TRAIN,
        root='./data/benchmark/atari',
        y_label='Average return',
        save_file='images/Breakout.png'
    )


if __name__ == '__main__':
    mkdir('images')
    plot_mujoco()
    plot_atari()