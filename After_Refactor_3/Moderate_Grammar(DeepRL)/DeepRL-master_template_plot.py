class PlotGenerator:
    def __init__(self, games, patterns, labels, root, plot_tag, agg='mean', downsample=0, right_align=False, interpolation=0, window=0):
        self.plotter = Plotter()
        self.games = games
        self.patterns = patterns
        self.labels = labels
        self.root = root
        self.plot_tag = plot_tag
        self.agg = agg
        self.downsample = downsample
        self.right_align = right_align
        self.interpolation = interpolation
        self.window = window
        self.image_name = f'images/{plot_tag}.png'

    def generate_plot(self):
        self.plotter.plot_games(games=self.games,
                                patterns=self.patterns,
                                agg=self.agg,
                                downsample=self.downsample,
                                labels=self.labels,
                                right_align=self.right_align,
                                tag=self.plot_tag,
                                root=self.root,
                                interpolation=self.interpolation,
                                window=self.window,
                                )
        plt.tight_layout()
        plt.savefig(self.image_name, bbox_inches='tight')
        plt.close()

RETURN_TRAIN = 'return_train'
RETURN_TEST = 'return_test'

MUJOCO_ROOT = './data/benchmark/mujoco'
ATARI_ROOT = './data/benchmark/atari'

PPO = 'PPO'
DDPG = 'DDPG'
TD3 = 'TD3'
A2C = 'A2C'
C51 = 'C51'
DQN = 'DQN'
N_STEP_DQN = 'N-Step DQN'
OC = 'OC'
QR_DQN = 'QR-DQN'

BATCH_SIZE = 100

def plot_ppo():
    generator = PlotGenerator(games=['HalfCheetah-v2','Walker2d-v2','Hopper-v2','Swimmer-v2','Reacher-v2','Ant-v2','Humanoid-v2','HumanoidStandup-v2'],
                              patterns=['remark_ppo'],
                              labels=[PPO],
                              root=MUJOCO_ROOT,
                              plot_tag=RETURN_TRAIN,
                              agg='mean',
                              downsample=0,
                              right_align=False,
                              interpolation=100,
                              window=10,
                              )
    generator.generate_plot()

def plot_ddpg_td3():
    generator = PlotGenerator(games=['HalfCheetah-v2','Walker2d-v2','Hopper-v2','Swimmer-v2','Reacher-v2','Ant-v2'],
                              patterns=['remark_ddpg', 'remark_td3'],
                              labels=[DDPG, TD3],
                              root=MUJOCO_ROOT,
                              plot_tag=RETURN_TEST,
                              agg='mean',
                              downsample=0,
                              right_align=False,
                              interpolation=0,
                              window=0,
                              )
    generator.generate_plot()

def plot_atari():
    generator = PlotGenerator(games=['BreakoutNoFrameskip-v4'],
                              patterns=['remark_a2c','remark_categorical','remark_dqn','remark_n_step_dqn','remark_option_critic','remark_quantile','remark_ppo'],
                              labels=[A2C, C51, DQN, N_STEP_DQN, OC, QR_DQN, PPO],
                              root=ATARI_ROOT,
                              plot_tag=RETURN_TRAIN,
                              agg='mean',
                              downsample=BATCH_SIZE,
                              right_align=False,
                              interpolation=0,
                              window=BATCH_SIZE,
                              )
    generator.generate_plot()
