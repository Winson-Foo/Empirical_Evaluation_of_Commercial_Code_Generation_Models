from typing import Dict
import torch

from .config import Config
from .utils import get_class_attr


class Trainer:
    """Trainer class for training the agents"""

    def __init__(self, config: Config, logger) -> None:
        """Construct the Trainer object"""
        self.config = config
        self.logger = logger
        self.logger.setLevel(config.log_level)

        self.task_fn = get_class_attr(config.task_fn)
        self.optimizer_fn = get_class_attr(config.optimizer_fn)
        self.actor_optimizer_fn = get_class_attr(config.actor_optimizer_fn)
        self.critic_optimizer_fn = get_class_attr(config.critic_optimizer_fn)
        self.network_fn = get_class_attr(config.network_fn)
        self.actor_network_fn = get_class_attr(config.actor_network_fn)
        self.critic_network_fn = get_class_attr(config.critic_network_fn)
        self.replay_fn = get_class_attr(config.replay_fn)
        self.random_process_fn = get_class_attr(config.random_process_fn)

        # Network initialization
        self.actor_network = self.actor_network_fn(
            self.config.state_dim, self.config.action_dim
        ).to(self.config.DEVICE)
        self.critic_network = self.critic_network_fn(
            self.config.state_dim, self.config.action_dim
        ).to(self.config.DEVICE)
        self.target_actor_network = self.actor_network_fn(
            self.config.state_dim, self.config.action_dim
        ).to(self.config.DEVICE)
        self.target_critic_network = self.critic_network_fn(
            self.config.state_dim, self.config.action_dim
        ).to(self.config.DEVICE)
        self.actor_optimizer = self.actor_optimizer_fn(
            self.actor_network.parameters(), lr=self.config.actor_lr
        )
        self.critic_optimizer = self.critic_optimizer_fn(
            self.critic_network.parameters(), lr=self.config.critic_lr
        )

        # Replay memory initialization
        self.memory = self.replay_fn(self.config.replay_size)

        # Random process initialization
        self.random_process = self.random_process_fn(
            self.config.action_dim, sigma=self.config.random_process_sigma
        )

        # Initialize counters
        self.total_steps = 0
        self.episode_rewards = []

    def train(self) -> None:
        """Train the agent"""
        state = self.task_fn.reset()
        episode_reward = 0
        self.episode_rewards.append(episode_reward)

        for step_i in range(self.config.max_steps):

            # Exploration vs exploitation
            random_action_prob = self.config.random_action_prob
            if random_action_prob is None:
                remaining_steps = self.config.max_steps - self.total_steps
                random_action_prob = self.config.random_action_prob_schedule(
                    remaining_steps
                )
                    
            action = self.act(state, random_action_prob)
            next_state, reward, done, _ = self.task_fn.step(action)

            # Add transition to replay memory
            self.memory.add(state, action, reward, next_state, done)

            # Update the network after every few steps
            if len(self.memory) > self.config.batch_size:
                experiences = self.memory.sample(self.config.batch_size)
                self.update_network(experiences)

            # Update the target network with the current network
            if (
                self.total_steps % self.config.target_network_update_freq == 0
            ):
                self.update_target_network()

            # Update counters
            self.total_steps += 1
            episode_reward += reward

            if done:
                state = self.task_fn.reset()
                self.episode_rewards.append(episode_reward)
                episode_reward = 0

            else:
                state = next_state

            if (
                len(self.episode_rewards) >= self.config.num_episodes_to_keep
                and self.config.terminate_on_solve
                and self.is_solved()
            ):
                self.logger.info(f"Solved in {self.total_steps} steps")
                break

            if self.total_steps % self.config.log_interval == 0:
                self.logger.debug(
                    f"Total steps: {self.total_steps} | Episode: {len(self.episode_rewards)} | Episode reward: {episode_reward}"
                )

    def act(self, state: torch.Tensor, random_action_prob: float) -> torch.Tensor:
        """Get action from actor network"""
        self.actor_network.eval()
        with torch.no_grad():
            action = self.actor_network(state.to(self.config.DEVICE)).cpu()
        self.actor_network.train()

        if torch.rand(1) < random_action_prob:
            action += torch.tensor(
                self.random_process.sample(), dtype=torch.float32
            )

        return action

    def update_network(self, experiences: Dict[str, torch.Tensor]) -> None:
        """Update the actor and critic networks"""
        self.actor_network.train()
        self.critic_network.train()

        states = experiences["states"].to(self.config.DEVICE)
        actions = experiences["actions"].to(self.config.DEVICE)
        rewards = experiences["rewards"].to(self.config.DEVICE)
        next_states = experiences["next_states"].to(self.config.DEVICE)
        dones = experiences["dones"].to(self.config.DEVICE)

        # Actor loss
        actions_pred = self.actor_network(states)
        actor_loss = -self.critic_network(states, actions_pred).mean()

        # Critic loss
        actions_next = self.target_actor_network(next_states)
        q_next = self.target_critic_network(next_states, actions_next)
        q_target = rewards + self.config.discount_factor * q_next * (1 - dones)
        q_pred = self.critic_network(states, actions)
        critic_loss = (q_pred - q_target.detach()).pow(2).mean()

        # Update the networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor_network.parameters(), self.config.max_grad_norm
        )
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic_network.parameters(), self.config.max_grad_norm
        )
        self.critic_optimizer.step()

    def update_target_network(self) -> None:
        """Update the target actor and critic networks with the current actor and critic networks"""
        self.target_actor_network.load_state_dict(
            self.config.target_network_mix * self.actor_network.state_dict()
            + (1 - self.config.target_network_mix)
            * self.target_actor_network.state_dict()
        )

        self.target_critic_network.load_state_dict(
            self.config.target_network_mix * self.critic_network.state_dict()
            + (1 - self.config.target_network_mix)
            * self.target_critic_network.state_dict()
        )

    def is_solved(self) -> bool:
        """Check if the problem is solved"""
        return (
            sum(self.episode_rewards[-self.config.num_episodes_to_keep :])
            >= self.config.solve_score
        )