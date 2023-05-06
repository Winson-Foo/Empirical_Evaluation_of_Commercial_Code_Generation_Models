import logging
import os
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .misc import get_time_str


class Logger:
    def __init__(
        self, tag: Optional[str] = "default", log_level: Optional[int] = 0
    ) -> None:
        """
        Initializes a new Logger instance.

        Args:
            tag: A tag for the logger (default is 'default').
            log_level: The log level for the logger (default is 0).
        """
        self.vanilla_logger = logging.getLogger()
        self.vanilla_logger.setLevel(logging.INFO)

        self.log_dir = f"./tf_log/logger-{tag}-{get_time_str()}"
        if tag is not None:
            os.makedirs("./log", exist_ok=True)
            fh = logging.FileHandler(f"./log/{tag}-{get_time_str()}.txt")
            fh.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
            )
            fh.setLevel(logging.INFO)
            self.vanilla_logger.addHandler(fh)

        self.writer = None
        self.log_level = log_level
        self.all_steps = {}

    def _lazy_init_writer(self) -> None:
        """
        Initializes the SummaryWriter instance lazily.
        """
        if self.writer is None:
            self.writer = SummaryWriter(self.log_dir)

    @staticmethod
    def _to_numpy(value: Any) -> Any:
        """
        Converts a tensor to a numpy array.

        Args:
            value: The tensor to convert.

        Returns:
            The tensor as a numpy array.
        """
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach().numpy()
        return value

    def get_step(self, tag: str) -> int:
        """
        Returns the current step for a given tag and increments it.

        Args:
            tag: The tag to get the step for.

        Returns:
            The current step for the tag.
        """
        if tag not in self.all_steps:
            self.all_steps[tag] = 0
        step = self.all_steps[tag]
        self.all_steps[tag] += 1
        return step

    def add_scalar(
        self, tag: str, value: Any, step: Optional[int] = None, log_level: Optional[int] = 0
    ) -> None:
        """
        Adds a scalar value to TensorBoard.

        Args:
            tag: The tag for the scalar value.
            value: The scalar value to add.
            step: The step to add the scalar value at.
            log_level: The log level for the scalar value.

        Returns:
            None
        """
        self._lazy_init_writer()
        if log_level > self.log_level:
            return
        value = self._to_numpy(value)
        if step is None:
            step = self.get_step(tag)
        if np.isscalar(value):
            value = np.asarray([value])
        self.writer.add_scalar(tag, value, step)

    def add_histogram(
        self, tag: str, values: Any, step: Optional[int] = None, log_level: Optional[int] = 0
    ) -> None:
        """
        Adds a histogram to TensorBoard.

        Args:
            tag: The tag for the histogram.
            values: The values for the histogram.
            step: The step to add the histogram at.
            log_level: The log level for the histogram.

        Returns:
            None
        """
        self._lazy_init_writer()
        if log_level > self.log_level:
            return
        values = self._to_numpy(values)
        if step is None:
            step = self.get_step(tag)
        self.writer.add_histogram(tag, values, step)