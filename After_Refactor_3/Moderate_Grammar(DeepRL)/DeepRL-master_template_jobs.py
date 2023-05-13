from examples.games import atari
from examples.games import mujoco
from examples.algos import *

class Config:
    def __init__(self, game_name, algo_name):
        self.game_name = game_name
        self.algo_name = algo_name
        self.game_params = []
        self.algo_params = []

    def add_game_param(self, **kwargs):
        self.game_params.append(kwargs)
    
    def add_algo_param(self, **kwargs):
        self.algo_params.append(kwargs)

def batch_atari(config):
    game_module = atari
    algo_modules = [dqn_pixel, quantile_regression_dqn_pixel, categorical_dqn_pixel, rainbow_pixel, a2c_pixel, n_step_dqn_pixel, option_critic_pixel, ppo_pixel]

    for game_params in config.game_params:
        for algo_module in algo_modules:
            for algo_params in config.algo_params:
                params = {**game_params, **algo_params, 'remark':algo_module.__name__}
                algo_module.run(game_module, params)

def batch_mujoco(config):
    game_module = mujoco
    algo_modules = [ppo_continuous, ddpg_continuous, td3_continuous]

    for game_params in config.game_params:
        for algo_module in algo_modules:
            for algo_params in config.algo_params:
                params = {**game_params, **algo_params, 'remark':algo_module.__name__}
                algo_module.run(game_module, params)

if __name__ == '__main__':
    config = Config('BreakoutNoFrameskip-v4', 'dqn_pixel')
    config.add_game_param(run=1)
    config.add_algo_param()

    batch_atari(config)

    config = Config('Humanoid-v2', 'ppo_continuous')
    config.add_game_param()
    config.add_algo_param(epochs=100)

    batch_mujoco(config)