import torch
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Any
from torch.utils.tensorboard import SummaryWriter
import os.path as osp
import logging
import pickle
import torch.multiprocessing as mp

from ..utils import close_obj, get_logger, mkdir
from ..tasks import BaseTask
from ..networks import BaseNetwork


class BaseAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(tag=config.tag, log_level=config.log_level)
        self.task_ind = 0
        self.total_steps = 0
        self.stats_filename = osp.join(config.log_dir, 'stats.pkl')
        self.model_filename = osp.join(config.log_dir, 'model.pt')
        self.writer = SummaryWriter(config.log_dir)
        
    def save(self):
        torch.save(self.network.state_dict(), self.model_filename)
        with open(self.stats_filename, 'wb') as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)

    def load(self):
        state_dict = torch.load(self.model_filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open(self.stats_filename, 'rb') as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))

    def eval_step(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        while True:
            action = self.eval_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            if ret is not None:
                break
        return ret

    def eval_episodes(self):
        episodic_returns = []
        for ep in range(self.config.eval_episodes):
            total_rewards = self.eval_episode()
            episodic_returns.append(np.sum(total_rewards))
        mean_return, std_error = np.mean(episodic_returns), np.std(episodic_returns) / np.sqrt(len(episodic_returns))
        self.writer.add_scalar('episodic_return_test', mean_return, self.total_steps)
        self.logger.info(f"steps {self.total_steps}, episodic_return_test {mean_return:.2f} ({std_error:.2f})")
        return {'episodic_return_test': mean_return}

    def record_online_return(self, info: Dict[str, Any], offset:int =0) -> None:
        if isinstance(info, dict):
            ret = info['episodic_return']
            if ret is not None:
                self.writer.add_scalar('episodic_return_train', ret, self.total_steps + offset)
                self.logger.info(f"steps {self.total_steps + offset}, episodic_return_train {ret}")
        elif isinstance(info, tuple):
            for i, info_ in enumerate(info):
                self.record_online_return(info_, i)
        else:
            raise NotImplementedError

    def switch_task(self) -> None:
        config = self.config
        if not config.tasks:
            return
        segs = np.linspace(0, config.max_steps, len(config.tasks) + 1)
        if self.total_steps > segs[self.task_ind + 1]:
            self.task_ind += 1
            self.task = config.tasks[self.task_ind]
            self.states = self.task.reset()
            self.states = config.state_normalizer(self.states)

    def record_episode(self, dir: str, env: BaseTask) -> None:
        mkdir(dir)
        steps = 0
        state = env.reset()
        while True:
            self.record_obs(env, dir, steps)
            action = self.record_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            steps += 1
            if ret is not None:
                break

    def record_step(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def record_obs(self, env: BaseTask, dir: str, steps: int) -> None:
        env = env.env.envs[0]
        obs = env.render(mode='rgb_array')
        imsave(f"{dir}/{steps:04d}.png", obs)


class BaseActor(mp.Process):
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    NETWORK = 4
    CACHE = 5

    def __init__(self, config: Dict[str, Any]):
        mp.Process.__init__(self)
        self.config = config
        self.__pipe, self.__worker_pipe = mp.Pipe()

        self._state = None
        self._task = None
        self._network = None
        self._total_steps = 0
        self.__cache_len = 2

        if not config.async_actor:
            self.start = lambda: None
            self.step = self._sample
            self.close = lambda: None
            self._set_up()
            self._task = config.task_fn()

    def run(self):
        self._set_up()
        config = self.config
        self._task = config.task_fn()

        cache = deque([], maxlen=2)
        while True:
            op, data = self.__worker_pipe.recv()
            if op == self.STEP:
                if not len(cache):
                    cache.append(self._sample())
                    cache.append(self._sample())
                self.__worker_pipe.send(cache.popleft())
                cache.append(self._sample())
            elif op == self.EXIT:
                self.__worker_pipe.close()
                return
            elif op == self.NETWORK:
                self._network = data
            else:
                raise NotImplementedError

    def _transition(self):
        raise NotImplementedError

    def _set_up(self):
        pass

    def step(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        self.__pipe.send([self.STEP, None])
        return self.__pipe.recv()

    def close(self) -> None:
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

    def set_network(self, net: BaseNetwork) -> None:
        if not self.config.async_actor:
            self._network = net
        else:
            self.__pipe.send([self.NETWORK, net])


class DmControlActor(BaseActor):
    def _sample(self):
        transitions = []
        for _ in range(self.config.sgd_update_frequency):
            transition = self._transition()
            if transition is not None:
                transitions.append(transition)
        return transitions

    def _transition(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._network.eval()
        state = self._state
        if state is None:
            state = self._task.reset()
            state = self.config.state_normalizer(state)
        tensor_state = torch.from_numpy(np.array([state]))
        with torch.no_grad():
            action = self._network(tensor_state).detach().cpu().numpy()[0]
        action = np.clip(action, -1, 1)
        next_state, reward, done, info = self._task.step(action)
        next_state = self.config.state_normalizer(next_state)
        self._total_steps += 1
        if done:
            self._state = None
        else:
            self._state = next_state
        return state, action, reward, next_state


class DmControlAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.actor = DmControlActor(config)
        self.actor.set_network(self.network)

    def close(self):
        close_obj(self.actor)

    def _train(self) -> Dict[str, Any]:
        transitions = self.actor.step()
        batch = self.config.batch_fn(transitions)
        states = torch.from_numpy(batch.states)
        actions = torch.from_numpy(batch.actions)
        rewards = torch.from_numpy(batch.rewards)
        masks = torch.from_numpy(batch.masks)
        next_states = torch.from_numpy(batch.next_states)

        self.optimizer.zero_grad()
        loss = self.network.loss(states, actions, rewards, masks, next_states)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()

        grad_norm = np.sqrt(sum([p.grad.data.norm(2).item() ** 2 for p in self.network.parameters()]))
        self.writer.add_scalar('grad_norm', grad_norm, self.total_steps)

        if self.total_steps // self.config.target_network_update_freq> self.config.last_target_update / self.config.target_network_update_freq:
            self.target_network.load_state_dict(self.network.state_dict())
            self.config.last_target_update = self.total_steps
        self.record_online_return(transitions)
        self.total_steps += self.config.batch_size * self.config.num_workers
        
        return {"loss": loss.item()}  

    def save(self):
        super().save()
        self.config.save(osp.join(self.config.log_dir, 'config.pkl'))

    def learn(self) -> None:
        config = self.config
        self.network.train()
        self.target_network.train()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.batcher = config.batcher_fn(config.batch_size * config.num_workers, config.discount)

        while self.total_steps < config.max_steps:
            if config.use_linear_lr_decay:
                self._adjust_learning_rate(self.optimizer, self.total_steps, config)
            if config.use_linear_clip_decay:
                self._adjust_clip_param(self.config, self.total_steps)

            episodes = []
            for _ in range(config.num_workers):
                episode = self.actor.collect_one_episode()
                episodes.append(episode)
            self.batcher.buffer_data(episodes)
            if self.batcher.is_full():
                batch = self.batcher.get_batch()
                # Training networks
                stats = self._train()
                self.writer.add_scalar('loss', stats["loss"], self.total_steps)
                self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.total_steps)
                # Evaluate agent
                if self.total_steps % config.eval_frequency == 0:
                    self.eval_episodes()
                # Save agent
                if config.save_frequency and self.total_steps % config.save_frequency == 0:
                    self.save()    

    def _adjust_learning_rate(self, optimizer, step, config):
        fraction = min(float(step) / config.max_steps, 1.0)
        lr = config.lr_init + fraction * (config.lr_final - config.lr_init)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _adjust_clip_param(self, config, step):
        fraction = min(float(step) / config.max_steps, 1.0)
        clip_param = config.clip_param_init + fraction * (config.clip_param_final - config.clip_param_init)
        config.gradient_clip = clip_param