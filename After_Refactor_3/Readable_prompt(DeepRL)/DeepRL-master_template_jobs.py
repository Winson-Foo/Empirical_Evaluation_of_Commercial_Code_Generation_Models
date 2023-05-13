from examples import *

def batch_atari():
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    games = ['BreakoutNoFrameskip-v4']
    algos = [dqn_pixel, quantile_regression_dqn_pixel, categorical_dqn_pixel, rainbow_pixel,
             a2c_pixel, n_step_dqn_pixel, option_critic_pixel, ppo_pixel]

    params = []
    for game in games:
        for r in range(1):
            for algo in algos:
                params.append([algo, dict(game=game, run=r, remark=algo.__name__)])

    algo, param = params[cf.i]
    algo(**param)
    exit()


def batch_mujoco():
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    games = ['HalfCheetah-v2', 'Walker2d-v2', 'Swimmer-v2', 'Hopper-v2', 'Reacher-v2', 'Ant-v2', 
             'Humanoid-v2', 'HumanoidStandup-v2']
    params = []
    for game in games:
        if 'Humanoid' in game:
            algos = [ppo_continuous]
        else:
            algos = [ppo_continuous, ddpg_continuous, td3_continuous]
        for algo in algos:
            for r in range(5):
                params.append([algo, dict(game=game, run=r)])

    algo, param = params[cf.i]
    algo(**param, remark=algo.__name__)
    exit()


if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    random_seed()
    select_device(-1)
    batch_mujoco()