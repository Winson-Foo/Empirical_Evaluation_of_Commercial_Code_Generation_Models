def plot_games_common(plotter, games, patterns, labels, tag, root):
    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean',
                       downsample=0,
                       labels=labels,
                       right_align=False,
                       tag=tag,
                       root=root,
                       interpolation=100 if 'mujoco' in root else 0,
                       window=10 if 'mujoco' in root else 100,
                       )
    plt.tight_layout()
    plt.savefig(f'images/{tag}.png', bbox_inches='tight')

def plot_ppo():
    plotter = Plotter()
    games = ['HalfCheetah-v2', 'Walker2d-v2', 'Hopper-v2', 'Swimmer-v2', 'Reacher-v2', 'Ant-v2', 'Humanoid-v2', 'HumanoidStandup-v2']
    patterns = ['remark_ppo']
    labels = ['PPO']

    plot_games_common(plotter, games, patterns, labels, Plotter.RETURN_TRAIN, './data/benchmark/mujoco')

def plot_ddpg_td3():
    plotter = Plotter()
    games = ['HalfCheetah-v2', 'Walker2d-v2', 'Hopper-v2', 'Swimmer-v2', 'Reacher-v2', 'Ant-v2']
    patterns = ['remark_ddpg', 'remark_td3']
    labels = ['DDPG', 'TD3']

    plot_games_common(plotter, games, patterns, labels, Plotter.RETURN_TEST, './data/benchmark/mujoco')

def plot_atari():
    plotter = Plotter()
    games = ['BreakoutNoFrameskip-v4']
    patterns = ['remark_a2c', 'remark_categorical', 'remark_dqn', 'remark_n_step_dqn', 'remark_option_critic', 'remark_quantile', 'remark_ppo']
    labels = ['A2C', 'C51', 'DQN', 'N-Step DQN', 'OC', 'QR-DQN', 'PPO']

    plot_games_common(plotter, games, patterns, labels, Plotter.RETURN_TRAIN, './data/benchmark/atari')