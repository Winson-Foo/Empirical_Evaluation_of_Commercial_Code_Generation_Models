from examples import *

GAMES = [
    'HalfCheetah-v2',
    'Walker2d-v2',
    'Swimmer-v2',
    'Hopper-v2',
    'Reacher-v2',
    'Ant-v2',
    'Humanoid-v2',
    'HumanoidStandup-v2',
]

ALGOS = [
    ppo_continuous,
    ddpg_continuous,
    td3_continuous,
]

def configure():
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()
    random_seed()
    select_device(-1)


def run_mujoco():
    for i in range(len(GAMES)):
        game = GAMES[i]
        for algo in ALGOS:
            for r in range(5):
                params = {
                    'game': game,
                    'run': r,
                }
                
                if 'Humanoid' in game:
                    remark = 'ppo_continuous'
                else:
                    remark = algo.__name__
                
                algo(**params, remark=remark)


def main():
    configure()
    mkdir('log')
    mkdir('data')

    run_mujoco()


if __name__ == '__main__':
    main()