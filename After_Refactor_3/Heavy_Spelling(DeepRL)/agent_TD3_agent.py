from torch.nn import functional as F
import numpy as np
import torch

def train_step(agent):
    if agent.state is None:
        agent.random_process.reset_states()
        agent.state = agent.task.reset()
        agent.state = agent.config.state_normalizer(agent.state)

    if agent.total_steps < agent.config.warm_up:
        action = [agent.task.action_space.sample()]
    else:
        action = agent.network(agent.state)
        action = to_np(action)
        action += agent.random_process.sample()
    action = np.clip(action, agent.task.action_space.low, agent.task.action_space.high)
    next_state, reward, done, info = agent.task.step(action)
    next_state = agent.config.state_normalizer(next_state)
    agent.record_online_return(info)
    reward = agent.config.reward_normalizer(reward)

    agent.replay.feed(dict(
        state=agent.state,
        action=action,
        reward=reward,
        next_state=next_state,
        mask=1-np.asarray(done, dtype=np.int32),
    ))

    if done[0]:
        agent.random_process.reset_states()
    agent.state = next_state
    agent.total_steps += 1

    if agent.total_steps >= agent.config.warm_up:
        train_agent(agent)

def train_agent(agent):
    transitions = agent.replay.sample()
    states = tensor(transitions.state)
    actions = tensor(transitions.action)
    rewards = tensor(transitions.reward).unsqueeze(-1)
    next_states = tensor(transitions.next_state)
    mask = tensor(transitions.mask).unsqueeze(-1)

    a_next = agent.target_network(next_states)
    noise = torch.randn_like(a_next).mul(agent.config.td3_noise)
    noise = noise.clamp(-agent.config.td3_noise_clip, agent.config.td3_noise_clip)

    min_a = float(agent.task.action_space.low[0])
    max_a = float(agent.task.action_space.high[0])
    a_next = (a_next + noise).clamp(min_a, max_a)

    q_1, q_2 = agent.target_network.q(next_states, a_next)
    target = rewards + agent.config.discount * mask * torch.min(q_1, q_2)
    target = target.detach()

    q_1, q_2 = agent.network.q(states, actions)
    critic_loss = F.mse_loss(q_1, target) + F.mse_loss(q_2, target)

    agent.network.zero_grad()
    critic_loss.backward()
    agent.network.critic_opt.step()

    if agent.total_steps % agent.config.td3_delay:
        action = agent.network(states)
        policy_loss = -agent.network.q(states, action)[0].mean()

        agent.network.zero_grad()
        policy_loss.backward()
        agent.network.actor_opt.step()

        soft_update(agent.target_network, agent.network)

def eval_step(agent, state):
    agent.config.state_normalizer.set_read_only()
    state = agent.config.state_normalizer(state)
    action = agent.network(state)
    agent.config.state_normalizer.unset_read_only()
    return to_np(action)

def soft_update(target, src, mix_ratio):
    for target_param, param in zip(target.parameters(), src.parameters()):
        target_param.detach_()
        target_param.copy_(target_param * (1.0 - mix_ratio) +
                           param * mix_ratio)

class TD3Agent:
    def __init__(self, config):
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None

    def step(self):
        train_step(self)