# Improved and Refactored Code:

# import statements
import torch.optim.lr_scheduler as lr_scheduler
from ..network import *
from ..component import *
from .BaseAgent import *

class PPOAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config) # calling BaseAgent constructor
        self.task = config.task_fn()
        self.network = config.network_fn()
        if config.shared_repr:
            self.optimizer = config.optimizer_fn(self.network.parameters())
        else:
            self.actor_optimizer = config.actor_opt_fn(self.network.actor_params)
            self.critic_optimizer = config.critic_opt_fn(self.network.critic_params)
        self.total_steps = 0
        self.states = config.state_normalizer(self.task.reset())
        if config.shared_repr:
            self.lr_scheduler = lr_scheduler.LambdaLR(self.optimizer, 
                                                      lambda step: 1 - step / config.max_steps)
        # initializing placeholders for tracking metrics during training
        self.all_ep_return = []
        self.max_ep_return = []
        self.min_ep_return = []
        self.mean_ep_return = []

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        # generating rollout
        for _ in range(config.rollout_length):
            prediction = self.network(states)
            next_states, rewards, terminals, info = self.task.step(prediction['action'].detach().numpy())
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            storage.add(prediction)
            storage.add({'reward': tensor(rewards).unsqueeze(-1),
                          'mask': tensor(1 - terminals).unsqueeze(-1),
                          'state': tensor(states)})
            states = next_states
            self.total_steps += config.num_workers
        # final state forwarding through network
        self.states = states
        prediction = self.network(states)
        storage.add(prediction)
        storage.placeholder()
        
        # calculating advantages and returns for each rollouts
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.reward[i] + config.discount * storage.mask[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.reward[i] + config.discount * storage.mask[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.mask[i] + td_error
            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        # extracting and normalizing the data from each rollouts
        entries = storage.extract(['state', 'action', 'log_pi_a', 'ret', 'advantage'])
        EntryCLS = entries.__class__
        entries = EntryCLS(*map(tensor, entries))
        entries.advantage = (entries.advantage - entries.advantage.mean()) / entries.advantage.std()
        
        # training the network using the extracted data
        if config.shared_repr:
            self.lr_scheduler.step(self.total_steps) # changing the learning rate of optimizer for shared representation
        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(entries.state.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                entry = EntryCLS(*map(lambda x: x[batch_indices], entries))
                
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
                    (policy_loss + value_loss).backward() # combined loss (actor critic loss)
                    nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                    self.optimizer.step()
                else:
                    if approx_kl <= 1.5 * config.target_kl:
                        self.actor_optimizer.zero_grad()
                        policy_loss.backward()
                        self.actor_optimizer.step()
                    self.critic_optimizer.zero_grad()
                    value_loss.backward()
                    self.critic_optimizer.step()
                    
        # storing metrics (online rewards of all worker threads) recorded during the current step
        self.all_ep_return.append(self.ep_return)
        self.max_ep_return.append(np.max(self.ep_return))
        self.min_ep_return.append(np.min(self.ep_return))
        self.mean_ep_return.append(np.mean(self.ep_return)) 

    # method for resetting rewards and 
    def reset_metrics(self):
        self.all_ep_return = []
        self.max_ep_return = []
        self.min_ep_return = []
        self.mean_ep_return = []
        super().reset_metrics()
        
    # method for returning a dict containing all the metrics tracked during this run of agent
    def get_metrics(self):
        metrics = super().get_metrics()
        metrics.update({'mean_ep_return' : np.mean(self.mean_ep_return),
                       'max_ep_return' : np.max(self.max_ep_return),
                       'min_ep_return' : np.min(self.min_ep_return),
                       'global_step' : self.total_steps})                  
        return metrics