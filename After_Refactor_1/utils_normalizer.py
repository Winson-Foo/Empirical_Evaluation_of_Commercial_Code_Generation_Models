import numpy as np
import torch
from typing import List, Dict


class BaseNormalizer:
    """
    Base class for normalizers.
    """
    def __init__(self, read_only: bool = False) -> None:
        """
        Initializes the BaseNormalizer.

        Args:
            read_only (bool): Whether the normalizer is read-only or not.
        """
        self._read_only = read_only

    def set_read_only(self) -> None:
        """
        Sets the read-only flag of the normalizer.
        """
        self._read_only = True

    def unset_read_only(self) -> None:
        """
        Unsets the read-only flag of the normalizer.
        """
        self._read_only = False

    def state_dict(self) -> Dict[str, np.ndarray]:
        """
        Returns the state dictionary of the normalizer.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        """
        Loads the state dictionary into the normalizer.

        Args:
            state_dict (Dict[str, np.ndarray]): The state dictionary to be loaded.
        """
        pass


class MeanStdNormalizer(BaseNormalizer):
    """
    Normalizes the input data by subtracting the mean and dividing by the standard deviation.

    Attributes:
        _rms (RunningMeanStd): The running mean and standard deviation of the input data.
        _clip (float): The maximum absolute value allowed after normalization.
        _epsilon (float): A small constant added to the standard deviation to prevent division by zero.

    """
    def __init__(self, read_only: bool = False, clip: float = 10.0, epsilon: float = 1e-8) -> None:
        """
        Initializes the MeanStdNormalizer.

        Args:
            read_only (bool): Whether the normalizer is read-only or not.
            clip (float): The maximum absolute value allowed after normalization.
            epsilon (float): A small constant added to the standard deviation to prevent division by zero.
        """
        super().__init__(read_only=read_only)
        self._rms = None
        self._clip = clip
        self._epsilon = epsilon

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Normalizes the input data.

        Args:
            x (np.ndarray): The input data to be normalized.

        Returns:
            np.ndarray: The normalized data.
        """
        x = np.asarray(x)
        if self._rms is None:
            self._rms = RunningMeanStd(shape=(1,) + x.shape[1:])
        if not self._read_only:
            self._rms.update(x)
        return np.clip((x - self._rms.mean) / np.sqrt(self._rms.var + self._epsilon),
                       -self._clip, self._clip)

    def state_dict(self) -> Dict[str, np.ndarray]:
        """
        Returns the state dictionary of the normalizer.
        """
        return {'mean': self._rms.mean,
                'var': self._rms.var}

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        """
        Loads the state dictionary into the normalizer.

        Args:
            state_dict (Dict[str, np.ndarray]): The state dictionary to be loaded.
        """
        self._rms.mean = state_dict['mean']
        self._rms.var = state_dict['var']


class RescaleNormalizer(BaseNormalizer):
    """
    Scales the input data by a constant coefficient.

    Attributes:
        _coef (float): The scaling coefficient.
    """
    def __init__(self, coef: float = 1.0) -> None:
        """
        Initializes the RescaleNormalizer.

        Args:
            coef (float): The scaling coefficient.
        """
        super().__init__()
        self._coef = coef

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Scales the input data.

        Args:
            x (np.ndarray): The input data to be scaled.

        Returns:
            np.ndarray: The scaled data.
        """
        if not isinstance(x, torch.Tensor):
            x = np.asarray(x)
        return self._coef * x


class ImageNormalizer(RescaleNormalizer):
    """
    Normalizes the input images by scaling the pixel values to the [0, 1] range.
    """
    def __init__(self) -> None:
        """
        Initializes the ImageNormalizer.
        """
        super().__init__(coef=1.0 / 255)


class SignNormalizer(BaseNormalizer):
    """
    Normalizes the input data by taking the sign of each element.
    """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Normalizes the input data.

        Args:
            x (np.ndarray): The input data to be normalized.

        Returns:
            np.ndarray: The normalized data.
        """
        return np.sign(x)