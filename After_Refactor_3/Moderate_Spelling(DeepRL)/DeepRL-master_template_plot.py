import matplotlib.pyplot as plt
from deep_rl import Plotter


def plot_ppo():
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

    patterns = ['remark_ppo']

    labels = ['PPO']

    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean',
                       downsample=0,
                       labels=labels,
                       right_align=False,
                       tag=plotter.RETURN_TRAIN,
                       root='./data/benchmark/mujoco',
                       interpolation=100,
                       window=10,
                       )

    plt.tight_layout()
    plt.savefig('images/PPO.png', bbox_inches='tight')


def plot_ddpg_td3():
    plotter = Plotter()
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Swimmer-v2',
        'Reacher-v2',
        'Ant-v2',
    ]

    patterns = ['remark_ddpg', 'remark_td3']

    labels = ['DDPG', 'TD3']

    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean',
                       downsample=0,
                       labels=labels,
                       right_align=False,
                       tag=plotter.RETURN_TEST,
                       root='./data/benchmark/mujoco',
                       interpolation=0,
                       window=0,
                       )

    plt.tight_layout()
    plt.savefig('images/mujoco_eval.png', bbox_inches='tight')


def plot_atari():
    plotter = Plotter()
    games = ['BreakoutNoFrameskip-v4']

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

    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean',
                       downsample=100,
                       labels=labels,
                       right_align=False,
                       tag=plotter.RETURN_TRAIN,
                       root='./data/benchmark/atari',
                       interpolation=0,
                       window=100,
                       )

    plt.tight_layout()
    plt.savefig('images/Breakout.png', bbox_inches='tight')


if __name__ == '__main__':
    if not os.path.exists('images'):
        os.makedirs('images')

    plot_atari()
    plot_ddpg_td3()
    plot_ppo()