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

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return to_np(action)

    def step(self):
        self._warm_up()
        self._take_action()
        self._process_feedback()

    def _warm_up(self):
        if self.total_steps < self.config.warm_up:
            self._random_action()

    def _take_action(self):
        action = self._get_action()
        next_state, reward, done, info = self.task.step(action)
        self._record_online_return(info)
        self._process_feedback()

    def _get_action(self):
        action = self.network(self.state)
        action = to_np(action)
        action += self.random_process.sample()
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        return action

    def _random_action(self):
        action = [self.task.action_space.sample()]

    def _process_feedback(self):
        next_state = self.config.state_normalizer(next_state)
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

        if self.replay.size() >= self.config.warm_up:
            self._update_networks()

    def _update_networks(self):
        transitions = self.replay.sample()
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

        phi = self.network.feature(states)
        action = self.network.actor(phi)
        policy_loss = -self.network.critic(phi.detach(), action).mean()

        self.network.zero_grad()
        policy_loss.backward()
        self.network.actor_opt.step()

        self.soft_update(self.target_network, self.network)