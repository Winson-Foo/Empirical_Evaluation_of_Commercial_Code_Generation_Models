import matplotlib.pyplot as plt
from deep_rl import *
import os


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
    ]

    labels = [
        'PPO'
    ]

    plotter.plot_games(
        games=games,
        patterns=patterns,
        agg='mean',
        labels=labels,
        right_align=False,
        tag=plotter.RETURN_TRAIN,
        root='./data/benchmark/mujoco',
        interpolation=100,
        window=10,
    )

    plt.tight_layout()
    plt.savefig('images/mujoco.png', bbox_inches='tight')


def plot_mujoco_eval():
    plotter = Plotter()
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

    plotter.plot_games(
        games=games,
        patterns=patterns,
        agg='mean',
        labels=labels,
        right_align=False,
        tag=plotter.RETURN_TEST,
        root='./data/benchmark/mujoco',
        window=0,
    )

    plt.tight_layout()
    plt.savefig('images/mujoco_eval.png', bbox_inches='tight')


def plot_atari_games():
    plotter = Plotter()

    def plot_breakout():
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

        plotter.plot_games(
            games=games,
            patterns=patterns,
            agg='mean',
            labels=labels,
            right_align=False,
            tag=plotter.RETURN_TRAIN,
            root='./data/benchmark/atari',
            window=100,
        )

        plt.tight_layout()
        plt.savefig('images/Breakout.png', bbox_inches='tight')

    # Call individual game plot functions
    plot_breakout()


if __name__ == '__main__':
    os.makedirs('images', exist_ok=True)
    plot_mujoco()
    plot_mujoco_eval()
    plot_atari_games()