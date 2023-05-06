import logging
import os
from typing import Any, Dict
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(
        self, tag: str = "default", log_level: int = logging.INFO, log_dir: str = "./log"
    ) -> None:
        self.tag = tag
        self.log_level = log_level
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.logger = self.get_logger()

    def log(self, level: int, message: str) -> None:
        if level >= self.log_level:
            self.logger.log(level, message)

    def debug(self, message: str) -> None:
        self.log(logging.DEBUG, message)

    def info(self, message: str) -> None:
        self.log(logging.INFO, message)

    def warning(self, message: str) -> None:
        self.log(logging.WARNING, message)

    def error(self, message: str) -> None:
        self.log(logging.ERROR, message)

    def get_step(self, tag: str) -> int:
        if tag not in self.all_steps:
            self.all_steps[tag] = 0
        step = self.all_steps[tag]
        self.all_steps[tag] += 1
        return step

    def add_scalar(self, tag: str, value: float, step: int = None) -> None:
        self.lazy_init_writer()
        value = self.to_numpy(value)
        if np.isscalar(value):
            value = np.asarray([value])
        if step is None:
            step = self.get_step(tag)
        self.writer.add_scalar(tag, value, step)

    def add_histogram(self, tag: str, values: Any, step: int = None) -> None:
        self.lazy_init_writer()
        values = self.to_numpy(values)
        if step is None:
            step = self.get_step(tag)
        self.writer.add_histogram(tag, values, step)

    def get_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.tag)
        logger.setLevel(self.log_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s: %(message)s"
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        file_handler = logging.FileHandler(
            os.path.join(self.log_dir, f"{self.tag}-{get_time_str()}.txt")
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    @staticmethod
    def to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach().numpy()
        return value

    def lazy_init_writer(self) -> None:
        if self.writer is None:
            self.writer = SummaryWriter(self.log_dir)

    def close(self) -> None:
        self.writer.close()


class TrainLogger(Logger):
    def __init__(
        self,
        *args,
        checkpoint_dir: str = "./checkpoint",
        checkpoint_interval: int = 1000,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.best_reward = float("-inf")
        self.all_rewards = []
        self.steps_since_last_checkpoint = 0

    def log_reward(self, reward: float, step: int) -> None:
        self.all_rewards.append(reward)
        self.add_scalar("reward", reward, step)
        avg_reward = np.mean(self.all_rewards[-100:])
        self.add_scalar("avg_reward", avg_reward, step)

    def log_step(self, step: int, losses: Dict[str, float]) -> None:
        self.add_scalar("steps", step)
        for k, v in losses.items():
            self.add_scalar(k, v, step)

    def log_checkpoint(self, policy_net, step) -> None:
        if self.checkpoint_dir is None:
            return
        self.steps_since_last_checkpoint += 1
        if (
            self.steps_since_last_checkpoint >= self.checkpoint_interval
            or step == 0
            or self.all_rewards[-1] > self.best_reward
        ):
            model_path = os.path.join(
                self.checkpoint_dir,
                f"{self.tag}-{step}-{self.all_rewards[-1]:.2f}.pth",
            )
            torch.save(policy_net.state_dict(), model_path)
            self.info(f"Saved model to {model_path}")
            self.best_reward = max(self.best_reward, self.all_rewards[-1])
            self.steps_since_last_checkpoint = 0

    def close(self) -> None:
        super().close()
        self.log_checkpoint(None, 0)