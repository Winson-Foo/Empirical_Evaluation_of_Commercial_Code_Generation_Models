# module: main.py

from typing import List, Tuple, Dict, Any
import torch.multiprocessing as mp
import torch.nn.functional as F
from .actors import BaseActor, ActorMessage
from .agents import BaseAgent


class Config:
    pass   # TODO: define the configuration class


def train(config: Config) -> None:
    mp.set_start_method('spawn')

    # set up actor processes
    actors = []
    pipes = []
    for i in range(config.num_actors):
        pipe1, pipe2 = mp.Pipe()
        actor = config.actor_fn(config, pipe2)
        actor.start()
        pipes.append(pipe1)
        actors.append(actor)

    # set up agent and network
    agent = config.agent_fn(config)
    network = config.network_fn(config)
    agent.network = network

    # train loop
    for i in range(config.max_steps):
        transitions = []
        for pipe in pipes:
            pipe.send([ActorMessage.NETWORK, network])

        for pipe in pipes:
            transitions.extend(pipe.recv())

        batch = agent.config.batch_fn(transitions)
        loss = agent.update(batch)

        with agent.logger:
            agent.total_steps += agent.config.num_workers * agent.config.sgd_update_frequency
            if i % agent.config.log_freq == 0:
                agent.logger.info('steps %d, loss %.3f' % (agent.total_steps, loss))
                agent.logger.add_scalar('loss', loss, agent.total_steps)

            if i % agent.config.eval_freq == 0:
                agent.eval_episodes()

            if agent.config.save_freq and i % agent.config.save_freq == 0:
                agent.save('%s/%s.%d' % (agent.config.save_dir, agent.config.tag, i))

            agent.switch_task()

    # cleanup
    for pipe, actor in zip(pipes, actors):
        pipe.send([ActorMessage.EXIT, None])
        pipe.close()
        actor.join()