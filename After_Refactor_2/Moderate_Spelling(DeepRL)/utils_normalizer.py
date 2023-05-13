import numpy as np
from typing import Tuple
from torch import Tensor
from baselines.common.running_mean_std import RunningMeanStd


class _BaseNormalizer:
    """
    Base class for normalizers
    """
    def __init__(self, read_only: bool = False):
        """
        Initializes the base normalizer

        :param read_only: If True, the normalizer is read only
        """
        self._read_only = read_only

    def set_read_only(self) -> None:
        """
        Sets the normalizer to be read only
        """
        self._read_only = True

    def unset_read_only(self) -> None:
        """
        Sets the normalizer to be writable
        """
        self._read_only = False

    def state_dict(self) -> None:
        """
        Returns the state dictionary of the normalizer
        """
        return None

    def load_state_dict(self, _state_dict: dict) -> None:
        """
        Loads the state dictionary of the normalizer

        :param _state_dict: The state dictionary to load
        """
        pass


class MeanStdNormalizer(_BaseNormalizer):
    """
    Normalizer using running mean and standard deviation
    """
    def __init__(self, read_only: bool = False, clip: float = 10.0,
                 epsilon: float = 1e-8) -> None:
        """
        Initializes the mean-std normalizer

        :param read_only: If True, the normalizer is read only
        :param clip: Value to clip the normalization
        :param epsilon: A small value to avoid divide-by-zero errors
        """
        super().__init__(read_only)
        self._rms = None
        self._clip = clip
        self._epsilon = epsilon

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Normalizes the input array using running mean and standard deviation

        :param x: The input array to normalize
        """
        x = np.asarray(x)
        if self._rms is None:
            self._rms = RunningMeanStd(shape=(1,) + x.shape[1:])
        if not self._read_only:
            self._rms.update(x)
        return np.clip((x - self._rms.mean) / np.sqrt(self._rms.var + self._epsilon),
                       -self._clip, self._clip)

    def state_dict(self) -> dict:
        """
        Returns the state dictionary of the normalizer
        """
        return {'mean': self._rms.mean,
                'var': self._rms.var}

    def load_state_dict(self, saved: dict) -> None:
        """
        Loads the state dictionary of the normalizer

        :param saved: The state dictionary to load
        """
        self._rms.mean = saved['mean']
        self._rms.var = saved['var']


class RescaleNormalizer(_BaseNormalizer):
    """
    Normalizer that rescales the input by a constant factor
    """
    def __init__(self, coef: float = 1.0) -> None:
        """
        Initializes the rescale normalizer

        :param coef: The constant factor to rescale by
        """
        super().__init__()
        self._coef = coef

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Rescales the input array by a constant factor

        :param x: The input array to rescale
        """
        if not isinstance(x, Tensor):
            x = np.asarray(x)
        return self._coef * x


class ImageNormalizer(RescaleNormalizer):
    """
    Normalizer that rescales RGB images by 1/255
    """
    def __init__(self) -> None:
        """
        Initializes the image normalizer
        """
        super().__init__(1.0 / 255)


class SignNormalizer(_BaseNormalizer):
    """
    Normalizer that takes the sign of the input
    """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Scales the input by its sign

        :param x: The input array
        """
        return np.sign(x)