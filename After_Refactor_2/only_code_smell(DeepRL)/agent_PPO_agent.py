from ..network import *
from ..component import *
from .BaseAgent import *


class PPOAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        
        self.network = config.network_fn()
        self.optimizer = (
            config.optimizer_fn(self.network.parameters())
            if config.shared_repr
            else (
                config.actor_opt_fn(self.network.actor_params),
                config.critic_opt_fn(self.network.critic_params)
            )
        )
        
        self.total_steps = 0
        self.states = self.config.state_normalizer(self.task.reset())
        
    def generate_rollouts(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        
        for i in range(config.rollout_length):
            prediction = self.network(states)
            action = to_np(prediction['action'])
            next_states, rewards, done, info = self.task.step(action)
            
            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            
            storage.feed(prediction)
            storage.feed({
                'reward': tensor(rewards).unsqueeze(-1),
                'mask': tensor(1 - done).unsqueeze(-1),
                'state': tensor(states)
            })
            
            states = next_states
            self.total_steps += config.num_workers
        
        self.states = states
        prediction = self.network(states)
        storage.feed(prediction)
        storage.placeholder()
        
        return storage
        
    def optimize(self, storage):
        config = self.config
        entries = storage.extract(['state', 'action', 'log_pi_a', 'ret', 'advantage'])
        EntryCLS = entries.__class__
        entries = EntryCLS(*list(map(lambda x: x.detach(), entries)))
        entries.advantage.copy_((entries.advantage - entries.advantage.mean()) / entries.advantage.std())
        
        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(entries.state.size(0)), config.mini_batch_size)
            
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                entry = EntryCLS(*list(map(lambda x: x[batch_indices], entries)))
                prediction = self.network(entry.state, entry.action)
                
                ratio = (prediction['log_pi_a'] - entry.log_pi_a).exp()
                obj = ratio * entry.advantage
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * entry.advantage
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['entropy'].mean()

                value_loss = 0.5 * (entry.ret - prediction['v']).pow(2).mean()

                approx_kl = (entry.log_pi_a - prediction['log_pi_a']).mean()
                
                if config.shared_repr:
                    self.optimizer.zero_grad()
                    (policy_loss + value_loss).backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                    self.optimizer.step()
                else:
                    if approx_kl <= 1.5 * config.target_kl:
                        self.optimizer[0].zero_grad()
                        policy_loss.backward()
                        self.optimizer[0].step()
                        
                    self.optimizer[1].zero_grad()
                    value_loss.backward()
                    self.optimizer[1].step()