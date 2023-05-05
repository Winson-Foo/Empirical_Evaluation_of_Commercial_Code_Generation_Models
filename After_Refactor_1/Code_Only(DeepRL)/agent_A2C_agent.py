from ..network import *
from ..component import *
from .BaseAgent import *


class A2CAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.task = config.task_fn()
        self.network = config.network_fn().to(config.device)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            prediction = self.network(config.state_normalizer(states))
            next_states, rewards, terminals, info = self.task.step(prediction['action'].cpu().detach().numpy())
            rewards = config.reward_normalizer(rewards)
            storage.add(prediction)
            storage.add({'reward': tensor(rewards).unsqueeze(-1),
                         'mask': tensor(1 - terminals).unsqueeze(-1)})
            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction = self.network(config.state_normalizer(states))
        storage.add(prediction)
        storage.placeholder()

        returns = prediction['v'].detach()
        advantages = tensor(np.zeros((config.num_workers, 1), dtype=np.float32))
        for i in reversed(range(config.rollout_length)):
            returns = storage.rewards[i] + config.discount * storage.masks[i] * returns
            if not config.use_gae:
                advantages = returns - storage.values[i].detach()
            else:
                td_error = storage.rewards[i] + config.discount * storage.masks[i] * storage.values[i + 1] - storage.values[i]
                advantages = advantages * config.gae_tau * config.discount * storage.masks[i] + td_error
            storage.advantages[i] = advantages.detach()
            storage.returns[i] = returns.detach()

        entries = storage.get()
        policy_loss = -(entries.log_probs * entries.advantages).mean()
        value_loss = F.mse_loss(entries.returns, entries.values)
        entropy_loss = entries.entropies.mean()

        self.optimizer.zero_grad()
        (policy_loss - config.entropy_weight * entropy_loss +
         config.value_loss_weight * value_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()