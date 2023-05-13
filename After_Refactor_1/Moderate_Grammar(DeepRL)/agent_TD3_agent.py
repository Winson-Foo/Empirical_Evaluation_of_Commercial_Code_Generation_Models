import torch
import torch.nn.functional as F
import numpy as np

from typing import Dict, Union
from gym.spaces import Space

from rlkit.torch.networks import Mlp, CNN
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.misc import ReplayBuffer, soft_update_from_to
from rlkit.torch.sac.agent import SAC


class TD3Agent(SAC):
    def __init__(
        self,
        obs_space: Space,
        action_space: Space,
        hidden_sizes: tuple[int] = (256, 256),
        cnn_params: Dict[str, Union[int, tuple]] = None,
        policy_class: type[TanhGaussianPolicy] = TanhGaussianPolicy,
        policy_kwargs: dict = None,
        replay_size: int = int(1e6),
        buffer_batch_size: int = 128,
        target_entropy: Union[str, float] = 'auto',
        discount: float = 0.99,
        soft_target_tau: float = 5e-3,
        policy_lr: float = 3e-4,
        qf_lr: float = 3e-4,
        reward_scale: Union[int, float] = 1,
        exploration_policy: bool = False,
        td3_delay: int = 2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
    ):
        super().__init__(
            obs_space,
            action_space,
            hidden_sizes,
            cnn_params,
            policy_class,
            policy_kwargs,
            replay_size,
            buffer_batch_size,
            target_entropy,
            discount,
            soft_target_tau,
            policy_lr,
            qf_lr,
            reward_scale,
            exploration_policy,
        )

        # TD3-specific hyperparameters
        self.td3_delay = td3_delay
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        # Initialize target Q functions
        self.qf2_target = self.qf2.copy()

    def train(self):
        # Sample a minibatch of transitions from replay buffer
        transitions = self.replay_buffer.sample(self.buffer_batch_size)
        obs, acts, rew, next_obs, done = (
            transitions['observations'],
            transitions['actions'],
            transitions['rewards'],
            transitions['next_observations'],
            transitions['terminals'],
        )

        with torch.no_grad():
            # Compute target actions with noise clipping
            target_policy_noise = torch.randn_like(acts) * self.noise_clip
            target_policy_noise = torch.clamp(
                target_policy_noise, -self.noise_clip, self.noise_clip
            )

            next_acts = self.policy(next_obs) + target_policy_noise
            next_acts = next_acts.clamp(self.action_space.low.min(), self.action_space.high.max())

            # Compute target Q-value estimates
            qf1_target_next, qf2_target_next = self.qf_target(next_obs, next_acts)
            min_qf_target_next = torch.min(qf1_target_next, qf2_target_next)
            target_q = rew + (1. - done) * self.discount * min_qf_target_next
            target_q = target_q.clamp(-self.reward_scale, 0)

        # Compute Q-function loss
        q1_pred, q2_pred = self.qf(obs, acts)
        qf1_loss = F.mse_loss(q1_pred, target_q)
        qf2_loss = F.mse_loss(q2_pred, target_q)
        qf_loss = qf1_loss + qf2_loss

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        # Delayed policy updates
        if self.total_steps % self.policy_delay == 0:
            policy_loss = self.qf(obs, self.policy(obs))[0].mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Soft update the target networks
            soft_update_from_to(self.qf, self.qf_target, self.soft_target_tau)
            soft_update_from_to(self.policy, self.policy_target, self.soft_target_tau)

            # TD3-specific updates
            if self.total_steps % self.td3_delay == 0:
                # Update Q-function targets
                self.qf2_target.load_state_dict(self.qf2.state_dict())

    def eval_action(self, obs):
        with torch.no_grad():
            obs = obs.unsqueeze(0)
            action = self.policy(obs)[0]
            return action.detach().cpu().numpy()