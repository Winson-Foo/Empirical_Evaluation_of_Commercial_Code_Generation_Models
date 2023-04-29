import numpy as np
import torch
from torch import tensor
from ..component import to_np

class DDPGConfig:
    warm_up = 10000
    target_network_mix = 0.001
    discount = 0.99

class DDPGAgent:
    """
    DDPG Agent class
    """

    def __init__(self, task_fn, network_fn, replay_fn, random_process_fn, config=DDPGConfig()):
        self.task = task_fn()  # task to perform
        self.network = network_fn()  # global network
        self.target_network = network_fn()  # target network

        # copy global network to target network
        self.target_network.load_state_dict(self.network.state_dict())

        self.replay_buffer = replay_fn()  # replay buffer
        self.random_process = random_process_fn()  # random process for exploration
        self.total_steps = 0  # total number of steps taken

        self.config = config
        self.state = None
        self.action_size = self.task.action_space.shape[0]

    def eval_step(self, state):
        """
        Get action for evaluation
        """
        state = self.config.state_normalizer(state)
        action = self.network(state)
        return to_np(action)

    def train(self, state, action, next_state, reward, mask):
        """
        Update network
        """
        phi_next = self.target_network.feature(next_state)
        a_next = self.target_network.actor(phi_next)
        q_next = self.target_network.critic(phi_next, a_next)
        q_next = self.config.discount * mask * q_next
        q_next.add_(reward)
        q_next = q_next.detach()

        phi = self.network.feature(state)
        q = self.network.critic(phi, action)
        critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()

        self.network.zero_grad()
        critic_loss.backward()
        self.network.critic_opt.step()

        phi = self.network.feature(state)
        action = self.network.actor(phi)
        policy_loss = -self.network.critic(phi.detach(), action).mean()

        self.network.zero_grad()
        policy_loss.backward()
        self.network.actor_opt.step()

        self.soft_update(self.target_network, self.network)

    def soft_update(self, target, src):
        """
        Soft update target network
        """
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) + param * self.config.target_network_mix)

    def step(self):
        """
        Take a step in the environment
        """
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = self.config.state_normalizer(self.state)

        if self.total_steps < self.config.warm_up:
            action = [self.task.action_space.sample()]
        else:
            action = self.network(self.state)
            action = to_np(action)
            action += self.random_process.sample()
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)

        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)

        self.replay_buffer.feed(dict(
            state=self.state,
            action=action,
            reward=reward,
            next_state=next_state,
            mask=1-np.asarray(done, dtype=np.int32),
        ))

        if done[0]:
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        if self.replay_buffer.size() >= self.config.warm_up:
            transitions = self.replay_buffer.sample()
            states = tensor(transitions.state)
            actions = tensor(transitions.action)
            rewards = tensor(transitions.reward).unsqueeze(-1)
            next_states = tensor(transitions.next_state)
            mask = tensor(transitions.mask).unsqueeze(-1)

            self.train(states, actions, next_states, rewards, mask)

    def record_online_return(self, info):
        """
        Record online returns
        """
        pass

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = 4
    action_dim = 2
    hidden_dim = 256

    task_fn = lambda : gym.make('CartPole-v0')
    network_fn = lambda: ActorCritic(state_dim, action_dim, hidden_dim).to(device)
    replay_fn = lambda: Replay(memory_size=int(1e5))
    random_process_fn = lambda: OrnsteinUhlenbeckProcess(size=action_dim, std=LinearSchedule(0.2))

    ddpg_agent = DDPGAgent(task_fn, network_fn, replay_fn, random_process_fn)
    while not ddpg_agent.task.finished:
        ddpg_agent.step()

if __name__ == '__main__':
    main()