# Refactored code:

# Import necessary libraries
import numpy as np
import torch
import torch.nn.functional as F

from rlkit.torch.networks import Mlp, ConcatMlp
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.data_management.replay_buffers.her_replay_buffer import HerReplayBuffer


class DDPGAgent:
    def __init__(self, env, task_fn, network_fn, replay_buffer_fn, state_normalizer, replay_buffer_size=int(1e6), **kwargs):
        self.env = env
        self.task_fn = task_fn
        self.network_fn = network_fn
        self.replay_buffer_fn = replay_buffer_fn
        self.state_normalizer = state_normalizer
        self.replay_buffer_size = replay_buffer_size
        self.random_process = kwargs.get('random_process', None)
        self.max_path_length = kwargs.get('max_path_length', 1000)
        self.policy = kwargs.get('policy', MakeDeterministic())
        self.reward_scale = kwargs.get('reward_scale', 1.0)
        self.buffer_batch_size = kwargs.get('buffer_batch_size', 256)
        self.use_her = kwargs.get('use_her', False)
        self.num_rollouts_per_horizon = kwargs.get('num_rollouts_per_horizon', 4)
        self.num_expl_envs = kwargs.get('num_expl_envs', 1)
        self.min_buffer_size = kwargs.get('min_buffer_size', int(1e4))
        self.warm_up_start_step = kwargs.get('warm_up_start_step', 1)
        self.warm_up_critic_only = kwargs.get('warm_up_critic_only', False)
        self.update_actor_every = kwargs.get('update_actor_every', 1)
        self.eval_env = kwargs.get('eval_env', None)
        self.eval_deterministic = kwargs.get('eval_deterministic', True)
        self.eval_num_episodes = kwargs.get('eval_num_episodes', 1)
        self.eval_max_path_length = kwargs.get('eval_max_path_length', None)
        self.callback = kwargs.get('callback', None)

        self._n_env_steps_total = 0
        self._n_rollouts_total = 0
        self._n_episodes_total = 0  # excludes rollouts that end because of horizon
        self._last_path_return = None
        self._last_path = None
        
        # Initialize task and networks
        self.task = self.task_fn()
        obs_space = self.env.observation_space
        action_space = self.env.action_space
        
        self.obs_dim = obs_space.low.size
        self.action_dim = action_space.low.size
        self.policy = self.network_fn(self.obs_dim, self.action_dim, **kwargs)
        self.target_policy = self.network_fn(self.obs_dim, self.action_dim, **kwargs)

        # Copy initial parameters from policy to target policy
        for target_param, source_param in zip(self.target_policy.parameters(), self.policy.parameters()):
            target_param.data.copy_(source_param.data)

        self.replay_buffer = self.replay_buffer_fn(self.replay_buffer_size, self.obs_dim, self.action_dim)

    def _train_one_step(self):
        with torch.no_grad():
            for _ in range(self.num_rollouts_per_horizon):
                obs = self.task.reset()
                self.policy.reset()
                self.target_policy.reset()
                obs = self.state_normalizer(obs)

                done = False
                while not done:
                    if self.random_process is not None:
                        action = np.clip(self.policy(obs) + self.random_process(), -1, 1)
                    else:
                        action = self.policy(obs)
                    action = np.clip(action, -1, 1)
                    action_pre = action.copy()
                    action = self.action_scaler(action)
                    new_obs, reward, done, info = self.task.step(action)
                    new_obs = self.state_normalizer(new_obs)
                    self._n_env_steps_total += 1
                    if isinstance(reward, float):
                        self._last_path_return += reward
                    else:
                        self._last_path_return += reward[0]

                    terminal = False
                    if self.max_path_length and self._n_env_steps_total >= self.max_path_length:
                        terminal = True
                        done = True

                    if self.use_her:
                        additional_reward = reward_fn(reward, new_obs, goal, achiev_goal)
                        reward = additional_reward
                        # add to replay buffer
                        self.replay_buffer.add(obs, action_pre, additional_reward, new_obs, terminal)
                    else:
                        reward *= self.reward_scale
                        self.replay_buffer.add(obs, action_pre, reward, new_obs, terminal)

                    obs = new_obs
                
                self._n_rollouts_total += 1
                if done:
                    self._n_episodes_total += 1

        self._last_path = dict(
            rewards=[self._last_path_return],
            actions=np.concatenate(self._last_actions),
            obs=np.stack(self._last_obs),
            next_obs=np.stack(self._last_next_obs),
            terminals=np.asarray([done]),
        )

        if self.replay_buffer.size() >= self.min_buffer_size:
            batch = np_to_pytorch_batch(self.replay_buffer.random_batch(self.buffer_batch_size))
            batch["observations"] = self.state_normalizer(batch["observations"])
            batch["next_observations"] = self.state_normalizer(batch["next_observations"])
            batch = self._preprocess_batch(batch)
            critic_loss, policy_loss = self._do_training(batch)
            self._update_targets()
        else:
            critic_loss = None
            policy_loss = None

        if critic_loss is not None:
            critic_loss = critic_loss.item()
        if policy_loss is not None:
            policy_loss = policy_loss.item()

        return {
            'Critic Loss': critic_loss,
            'Policy Loss': policy_loss,
            'Env Steps': self._n_env_steps_total,
            'Rollouts': self._n_rollouts_total,
            'Episodes': self._n_episodes_total,
        }

    def _do_training(self, batch):
        raise NotImplementedError()

    def _update_targets(self):
        for target_param, source_param in zip(self.target_policy.parameters(), self.policy.parameters()):
            target_param.data.copy_(self.target_policy_tau * source_param.data + (1 - self.target_policy_tau) * target_param.data)

    def _preprocess_batch(self, batch):
        return batch

    def evaluate(self, epoch):
        if self.eval_env is None:
            return False
        self.eval_env.reset()
        self.policy.reset()
        self.target_policy.reset()
        if self.eval_deterministic:
            ob, r, d, env_info = self.eval_env.step(np.zeros(self.action_dim))
        else:
            self.random_process.reset_states()
            ob, r, d, env_info = self.eval_env.step(self.eval_env.action_space.sample())
        ob = self.state_normalizer(ob)
        terminal = False
        path_length = 0
        eval_returns = []
        while not terminal and (self.eval_max_path_length is None or path_length < self.eval_max_path_length):
            if self.eval_deterministic:
                action = np.clip(self.policy(ob), -1, 1)
            else:
                action = self.policy(ob) + self.random_process()
                action = np.clip(action, -1, 1)
            next_ob, r, terminal, env_info = self.eval_env.step(action)
            next_ob = self.state_normalizer(next_ob)
            path_length += 1
            if isinstance(r, float):
                eval_returns.append(r)
            else:
                eval_returns.append(r[0])
            ob = next_ob
        mean_return = np.mean(eval_returns)
        mean_return = np.mean(mean_return) if isinstance(mean_return, np.ndarray) else mean_return
        self.eval_statistics['Epoch'] = epoch
        self.eval_statistics['Evaluation Success'] = int(env_info['success'])
        self.eval_statistics['Evaluation Returns'] = eval_returns
        self.eval_statistics['Evaluation Mean Return'] = mean_return
        self.eval_statistics['Path Length'] = path_length
        return True

    def save(self, epoch):
        torch.save(self.policy.state_dict(), '{}_policy.pth'.format(epoch))
        torch.save(self.target_policy.state_dict(), '{}_target_policy.pth'.format(epoch))
        self.replay_buffer.save()

    def load(self, epoch):
        self.policy.load_state_dict(torch.load('{}_policy.pth'.format(epoch)))
        self.target_policy.load_state_dict(torch.load('{}_target_policy.pth'.format(epoch)))
        self.replay_buffer.load()

    def __getstate__(self):
        state = dict(self.__dict__)
        state['policy_state_dict'] = self.policy.state_dict()
        state['target_policy_state_dict'] = self.target_policy.state_dict()
        del state['policy']
        del state['target_policy']
        del state['replay_buffer']
        del state['state_normalizer']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.policy = self.network_fn(self.obs_dim, self.action_dim)
        self.policy.load_state_dict(state['policy_state_dict'])
        self.target_policy = self.network_fn(self.obs_dim, self.action_dim)
        self.target_policy.load_state_dict(state['target_policy_state_dict'])
        self.replay_buffer = self.replay_buffer_fn(self.replay_buffer_size, self.obs_dim, self.action_dim)
        self.state_normalizer = state_normalizer


def ddpg_agent(*args, **kwargs):
    return DDPGAgent(*args, **kwargs)