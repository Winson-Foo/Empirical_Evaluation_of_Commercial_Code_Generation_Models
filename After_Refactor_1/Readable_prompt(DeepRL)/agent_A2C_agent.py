from ..network import *
from ..component import *
from .BaseAgent import *


class A2CAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()

    def step(self):
        config = self.config
        rollout_length = config.rollout_length
        num_workers = config.num_workers
        state_normalizer = config.state_normalizer
        reward_normalizer = config.reward_normalizer
        discount = config.discount
        use_gae = config.use_gae
        gae_tau = config.gae_tau
        entropy_weight = config.entropy_weight
        value_loss_weight = config.value_loss_weight
        gradient_clip = config.gradient_clip
        storage = Storage(rollout_length)
        states = self.states

        for _ in range(rollout_length):
            prediction = self.network(state_normalizer(states))
            actions = to_np(prediction['action'])
            next_states, rewards, terminals, info = self.task.step(actions)
            rewards = reward_normalizer(rewards)
            self.record_online_return(info)
            storage.feed(prediction)
            storage.feed({
                'reward': tensor(rewards).unsqueeze(-1),
                'mask': tensor(1 - terminals).unsqueeze(-1)
            })
            states = next_states
            self.total_steps += num_workers

        self.states = states
        prediction = self.network(state_normalizer(states))
        storage.feed(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((num_workers, 1)))
        returns = prediction['v'].detach()

        for i in reversed(range(rollout_length)):
            returns = storage.reward[i] + discount * storage.mask[i] * returns
            if not use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.reward[i] + discount * storage.mask[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * gae_tau * discount * storage.mask[i] + td_error
            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        entries = storage.extract(['log_pi_a', 'v', 'ret', 'advantage', 'entropy'])
        policy_loss = -(entries.log_pi_a * entries.advantage).mean()
        value_loss = 0.5 * (entries.ret - entries.v).pow(2).mean()
        entropy_loss = entries.entropy.mean()

        self.optimizer.zero_grad()
        (policy_loss - entropy_weight * entropy_loss +
         value_loss_weight * value_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), gradient_clip)
        self.optimizer.step()