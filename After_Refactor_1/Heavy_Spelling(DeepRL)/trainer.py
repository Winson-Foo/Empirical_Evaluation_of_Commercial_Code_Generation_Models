## trainer.py

from config import Config
from torch.optim import Adam
import torch.nn.functional as F

device = Config.DEVICE

class Trainer:
    def __init__(self, config):
        self.config = config

        # Initialize networks
        self.actor = config.actor_network_fn().to(device)
        self.critic = config.critic_network_fn().to(device)
        self.actor_target = config.actor_network_fn().to(device)
        self.critic_target = config.critic_network_fn().to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Initialize optimizers
        self.actor_optimizer = config.actor_optimizer_fn(self.actor.parameters())
        self.critic_optimizer = config.critic_optimizer_fn(self.critic.parameters())

        # Initialize replay buffer
        self.memory = config.replay_fn(config.min_memory_size)

    def train_step(self, state, action, next_state, reward, done):
        state = self.config.state_normalizer(state)
        next_state = self.config.state_normalizer(next_state)
        reward = self.config.reward_normalizer(reward)

        # Compute TD target
        target_q = self.critic_target(next_state, self.actor_target(next_state))
        td_target = reward + self.config.discount * target_q * (1 - done)

        current_q = self.critic(state, action)
        td_error = td_target - current_q

        # Critic loss
        critic_loss = (td_error ** 2).mean()

        # Actor loss
        actor_loss = -(self.critic(state, self.actor(state)).mean())

        # Entropy regularization
        entropy = -self.config.entropy_weight * (self.actor.log_prob(state)).mean()
        
        # Total loss
        loss = critic_loss + actor_loss + entropy

        # Update actor and critic
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        if self.config.gradient_clip is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # Update target networks
        self.update_targets()

        return critic_loss, actor_loss, entropy

    def update_targets(self):
        tau = self.config.target_network_mix
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def run(self):
        # Train loop
        for i in range(self.config.max_steps):
            # Collect experience
            state = self.config.eval_env.reset()
            episode_reward = 0
            for t in range(self.config.rollout_length):
                if self.config.random_action_prob is not None and np.random.uniform() < self.config.random_action_prob:
                    action = self.config.eval_env.sample_action()
                else:
                    action = self.actor(torch.tensor(state).to(device)).detach().cpu().numpy()
                next_state, reward, done, _ = self.config.eval_env.step(action)
                self.memory.add(Transition(state, action, reward, next_state, done))
                episode_reward += reward
                state = next_state

            if len(self.memory) < self.config.mini_batch_size:
                continue

            # Train network
            for j in range(self.config.optimization_epochs):
                batch = self.memory.sample(self.config.mini_batch_size)
                critic_loss, actor_loss, entropy = self.train_step(*batch)

            # Logging
            if i % self.config.log_interval == 0:
                print(f'Step {i}, Reward: {episode_reward:.2f}, Critic Loss: {critic_loss:.2f}, Actor Loss: {actor_loss:.2f}, Entropy: {entropy:.2f}')

            # Evaluation
            if self.config.eval_env is not None and i % self.config.eval_interval == 0:
                print(f'Evaluating at step {i}...')
                with torch.no_grad():
                    total_reward = 0
                    for _ in range(self.config.eval_episodes):
                        state = self.config.eval_env.reset()
                        done = False
                        while not done:
                            action = self.actor(torch.tensor(state).to(device)).detach().cpu().numpy()
                            next_state, reward, done, _ = self.config.eval_env.step(action)
                            state = next_state
                            total_reward += reward
                    average_reward = total_reward / self.config.eval_episodes
                    print(f'Average reward: {average_reward:.2f}') 

        # Save model
        if self.config.save_interval != 0 and (i+1) % self.config.save_interval == 0:
            torch.save(self.actor.state_dict(), f'actor_{i}.pth')
            torch.save(self.critic.state_dict(), f'critic_{i}.pth')