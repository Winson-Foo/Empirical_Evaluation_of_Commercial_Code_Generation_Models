from examples import *
from config import atari_games, mujoco_games, algos_atari, algos_mujoco


def batch_atari():
    """
    Runs Atari game experiments using the specified set of algorithms and game environments
    """
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    params = []
    
    # create a set of parameters for each combination of game and algorithm
    for game in atari_games:
        for r in range(1):
            for algo in algos_atari:
                params.append([algo, dict(game=game, run=r, remark=algo.__name__)])

    # select a set of parameters based on the provided index
    algo, param = params[cf.i]
    algo(**param)
    exit()


def batch_mujoco():
    """
    Runs Mujoco game experiments using the specified set of algorithms and game environments
    """
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    params = []

    # create a set of parameters for each combination of game and algorithm
    for game in mujoco_games:
        if 'Humanoid' in game:
            algos = [ppo_continuous]
        else:
            algos = algos_mujoco
        
        for algo in algos:
            for r in range(5):
                params.append([algo, dict(game=game, run=r)])

    # select a set of parameters based on the provided index
    algo, param = params[cf.i]
    algo(**param, remark=algo.__name__)

    exit()


if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    random_seed()

    # select_device(0)
    # batch_atari()

    select_device(-1)
    batch_mujoco()