# main.py
from src.network import *
from src.component import *
from src.utils import *
import time
from src.actor import *
from src.agent import *

def run(config):
    actor = DQNActor(config)
    agent = DQNAgent(config, actor)
    total_steps = 0
    while total_steps < config.max_steps:
        transitions = agent.step()
        total_steps += len(transitions)
        if total_steps >= config.exploration_steps:
            loss = agent.learn()