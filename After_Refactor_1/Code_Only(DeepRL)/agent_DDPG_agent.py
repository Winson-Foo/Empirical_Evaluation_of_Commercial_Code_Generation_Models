from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision


class DDPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return to_np(action)

    def step(self):
        if self.state is None:
            self.reset()
        else:
            self.update_learning()

    def reset(self):
        self.random_process.reset_states()
        self.state = self.task.reset()
        self.state = self.config.state_normalizer(self.state)
        self.take_action()

    def take_action(self):
        if self.total_steps < self.config.warm_up:
            self.take_random_action()
        else:
            self.take_network_action()

    def take_random_action(self):
        action = [self.task.action_space.sample()]
        self.replay.feed(dict(
            state=self.state,
            action=action,
            reward=0,
            next_state=self.state,
            mask=0
        ))
        self.state = None

    def take_network_action(self):
        action = self.network(self.state)
        action = to_np(action)
        action += self.random_process.sample()
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)
        self.replay.feed(dict(
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

    def update_learning(self):
        if self.replay.size() >= self.config.warm_up:
            transitions = self.replay.sample()
            self.update_critic(transitions)
            self.update_actor(transitions)
            self.soft_update(self.target_network, self.network)

    def update_critic(self, transitions):
        states = tensor(transitions.state)
        actions = tensor(transitions.action)
        rewards = tensor(transitions.reward).unsqueeze(-1)
        next_states = tensor(transitions.next_state)
        mask = tensor(transitions.mask).unsqueeze(-1)
        phi_next = self.target_network.feature(next_states)
        a_next = self.target_network.actor(phi_next)
        q_next = self.target_network.critic(phi_next, a_next)
        q_next = self.config.discount * mask * q_next
        q_next.add_(rewards)
        q_next = q_next.detach()
        phi = self.network.feature(states)
        q = self.network.critic(phi, actions)
        critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()
        self.network.zero_grad()
        critic_loss.backward()
        self.network.critic_opt.step()

    def update_actor(self, transitions):
        states = tensor(transitions.state)
        phi = self.network.feature(states)
        action = self.network.actor(phi)
        policy_loss = -self.network.critic(phi.detach(), action).mean()
        self.network.zero_grad()
        policy_loss.backward()
        self.network.actor_opt.step()

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) + param * self.config.target_network_mix)