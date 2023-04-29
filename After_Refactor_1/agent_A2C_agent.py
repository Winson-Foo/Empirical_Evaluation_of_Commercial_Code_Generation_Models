from ..network import *
from ..component import *
from .BaseAgent import *


class A2CAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.environment = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.environment.reset()

    def step(self):
        config = self.config
        storage = RolloutStorage(config.rollout_length, config.num_workers, 
                                  self.network.output_size, device=self.device)
        
        states = self.states
        for _ in range(config.rollout_length):
            predicted_values, predicted_actions, predicted_log_probs, predicted_entropies= self._forward(states)
            next_states, rewards, terminals, info = self.environment.step(to_np(predicted_actions))
            self._record_online_return(info)
            rewards = self._normalize_rewards(rewards)
            storage.feed(predicted_values, predicted_actions, predicted_log_probs, 
                          predicted_entropies, rewards, terminals)
            states = next_states
            self.total_steps += config.num_workers
        self.states = states

        predicted_values, predicted_actions, predicted_log_probs, predicted_entropies= self._forward(states)
        storage.feed(predicted_values, predicted_actions, predicted_log_probs, 
                      predicted_entropies, None, None)
        storage.calculate_returns(config.discount, config.use_gae, config.gae_tau)

        for i in range(config.ppo_num_epochs):
            for batch in storage.mini_batches(config.ppo_batch_size):
                values, actions, log_probs_old, entropies_old, \
                returns, advantages = self._fetch_minibatch(batch)
                values_clipped = values.detach() - \
                (values.detach() - returns).clamp(-config.ppo_clip_ratio, config.ppo_clip_ratio)
                value_loss = F.mse_loss(values, returns)
                advantages_clipped = advantages.detach() - \
                        (advantages.detach() - advantages.mean(dim=0)).clamp(-config.ppo_clip_ratio, config.ppo_clip_ratio)
                policy_loss = -(advantages * log_probs_old.exp()).mean(dim=0)
                clipped_policy_loss = -(advantages_clipped * log_probs_old.exp()).mean(dim=0)
                
                entropy_loss = entropies_old.mean(dim=0)
                loss = (clipped_policy_loss + value_loss * config.ppo_value_loss_weight
                        - entropy_loss * config.ppo_entropy_weight)
                self._optimize(loss)

    def _forward(self, states):
        predicted_values, predicted_actions, predicted_log_probs, predicted_entropies = self.network(states)
        return predicted_values, predicted_actions, predicted_log_probs, predicted_entropies

    def _record_online_return(self, info):
        if info[0].get('episode'):
            self.episode_rewards.append(info[0]['episode']['r'])

    def _normalize_rewards(self, rewards):
        return torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)

    def _fetch_minibatch(self, batch):
        states, actions, log_probs_old, entropies_old, returns, advantages = batch
        return states, actions, log_probs_old, entropies_old, returns, advantages

    def _optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()


class RolloutStorage:
    def __init__(self, rollout_length, num_workers, output_size, device='cpu'):
        self.rollout_length = rollout_length
        self.num_workers = num_workers
        self.device = device

        self.values = torch.zeros((rollout_length + 1, num_workers, 1), dtype=torch.float32, device=device)
        self.actions = torch.zeros((rollout_length, num_workers, output_size), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((rollout_length, num_workers, 1), dtype=torch.float32, device=device)
        self.entropies = torch.zeros((rollout_length, num_workers, 1), dtype=torch.float32, device=device)
        
        self.rewards = torch.zeros((rollout_length, num_workers, 1), dtype=torch.float32, device=device)
        self.terminals = torch.zeros((rollout_length, num_workers, 1), dtype=torch.uint8, device=device)
        self.returns = torch.zeros((rollout_length + 1, num_workers, 1), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((rollout_length, num_workers, 1), dtype=torch.float32, device=device)

        self.pos = 0
        
    def feed(self, values, actions, log_probs, entropies, rewards, terminals):
        self.values[self.pos + 1].copy_(values)
        self.actions[self.pos].copy_(actions)
        self.log_probs[self.pos].copy_(log_probs)
        self.entropies[self.pos].copy_(entropies)
        self.pos = (self.pos + 1) % self.rollout_length

        if rewards is not None:
            self.rewards[self.pos].copy_(rewards)

        if terminals is not None:
            self.terminals[self.pos].copy_(terminals)
        
    def calculate_returns(self, discount, use_gae, gae_tau):
        self.returns[-1].copy_(self.values[-1])
        self.advantages[...] = 0

        if use_gae:
            gae = 0
            for i in reversed(range(self.rollout_length)):
                delta = self.rewards[i] + discount * (1 - self.terminals[i]) * self.values[i + 1] - self.values[i] 
                gae = delta + discount * gae_tau * (1 - self.terminals[i]) * gae
                self.advantages[i].copy_(gae)
            
            self.returns[:-1].copy_(self.advantages + self.values[:-1])
        else:
            for i in reversed(range(self.rollout_length)):
                self.returns[i].copy_(self.returns[i + 1] * discount * (1 - self.terminals[i]) + self.rewards[i])

    def placeholder(self):
        self.values[0].copy_(self.values[-1])
    
    def mini_batches(self, batch_size):
        num_steps = self.rollout_length * self.num_workers
        indices = np.arange(num_steps)
        np.random.shuffle(indices)

        for i in range(num_steps // batch_size):
            mini_batch = indices[i * batch_size:(i + 1) * batch_size]
            yield (self.values[:-1].reshape(-1, 1)[mini_batch],
                   self.actions.reshape(num_steps, -1)[mini_batch],
                   self.log_probs.reshape(num_steps, 1)[mini_batch],
                   self.entropies.reshape(num_steps, 1)[mini_batch],
                   self.returns[:-1].reshape(-1, 1)[mini_batch],
                   self.advantages.reshape(-1, 1)[mini_batch])