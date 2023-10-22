from ..network import *
from ..component import *
from .BaseAgent import *


class A2CAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.task, self.network, self.optimizer, self.total_steps, self.states = config.task_fn(), \
                                                                                   config.network_fn(), \
                                                                                   config.optimizer_fn(self.network.parameters()), \
                                                                                   0, \
                                                                                   config.task_fn().reset()

    def step(self):
        config, storage, states = self.config, Storage(config.rollout_length), self.states
        for _ in range(config.rollout_length):
            prediction = self.network(config.state_normalizer(states))
            actions = to_np(prediction['action'])
            next_states, rewards, terminals, info = self.task.step(actions)
            rewards = config.reward_normalizer(rewards)
            self.record_online_return(info)
            storage.feed(prediction, tensor(rewards).unsqueeze(-1), tensor(1 - terminals).unsqueeze(-1))
            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction = self.network(config.state_normalizer(states))
        storage.feed(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['v'].detach()

        for i in reversed(range(config.rollout_length)):
            mask = config.discount * storage.mask[i]
            returns = storage.reward[i] + mask * returns
            td_error = storage.reward[i] + mask * storage.v[i + 1] - storage.v[i]
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                advantages = config.gae_tau * advantages * mask + td_error
            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        entries = storage.extract(['log_pi_a', 'v', 'ret', 'advantage', 'entropy'])
        policy_loss = - (entries.log_pi_a * entries.advantage).mean()
        value_loss = (entries.ret - entries.v).pow(2).mean() * 0.5
        entropy_loss = entries.entropy.mean()

        self.optimizer.zero_grad()
        (policy_loss - config.entropy_weight * entropy_loss +
         config.value_loss_weight * value_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()