from typing import Union, Optional, Any
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import eagerpy as ep

from ..devutils import atleast_kd
from ..models import Model
from ..criteria import Criterion
from ..distances import Distance
from .base import FlexibleDistanceMinimizationAttack
from .base import T
from .base import get_channel_axis
from .base import raise_if_kwargs
from .base import verify_input_bounds


class GaussianBlurAttack(FlexibleDistanceMinimizationAttack):
    """Blurs the inputs using a Gaussian filter with linearly
    increasing standard deviation.

    Args:
        steps : Number of sigma values tested between 0 and max_sigma.
        channel_axis : Index of the channel axis in the input data.
        max_sigma : Maximally allowed sigma value of the Gaussian blur.
    """

    def __init__(
        self,
        *,
        distance: Optional[Distance] = None,
        steps: int = 1000,
        channel_axis: Optional[int] = None,
        max_sigma: Optional[float] = None,
    ):
        super().__init__(distance=distance)
        self.steps = steps
        self.channel_axis = channel_axis
        self.max_sigma = max_sigma

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        verify_input_bounds(x, model)
        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)

        if x.ndim != 4:
            raise NotImplementedError(
                "only implemented for inputs with two spatial dimensions (and one channel and one batch dimension)"
            )

        channel_axis = self._get_channel_axis(model, x.ndim)
        self._validate_channel_axis(channel_axis)

        max_sigma = self._calculate_max_sigma(x, channel_axis)

        x0_ = x.numpy()
        result = x
        found = is_adversarial(x)
        epsilon = 0.0
        stepsize = 1.0 / self.steps

        for _ in range(self.steps):
            epsilon += stepsize
            sigmas = self._calculate_sigmas(x0_.ndim, channel_axis, epsilon, max_sigma)
            x_ = self._apply_gaussian_filter(x0_, sigmas)
            x_ = self._clip_values(x_, model)
            x = ep.from_numpy(x0, x_)
            is_adv = is_adversarial(x)
            new_adv = ep.logical_and(is_adv, found.logical_not())
            result = ep.where(atleast_kd(new_adv, x.ndim), x, result)
            found = ep.logical_or(new_adv, found)

            if found.all():
                break

        return restore_type(result)

    def _get_channel_axis(self, model: Model, ndim: int) -> Optional[int]:
        if self.channel_axis is None:
            return get_channel_axis(model, ndim)
        else:
            return self.channel_axis % ndim

    def _validate_channel_axis(self, channel_axis: Optional[int]) -> None:
        if channel_axis is None:
            raise ValueError(
                "cannot infer the data_format from the model, please specify"
                " channel_axis when initializing the attack"
            )

    def _calculate_max_sigma(self, x: T, channel_axis: int) -> float:
        if self.max_sigma is None:
            if channel_axis == 1:
                h, w = x.shape[2:4]
            elif channel_axis == 3:
                h, w = x.shape[1:3]
            else:
                raise ValueError(
                    f"expected 'channel_axis' to be 1 or 3, got {channel_axis}"
                )
            return max(h, w)
        else:
            return self.max_sigma

    def _calculate_sigmas(
        self, ndim: int, channel_axis: int, epsilon: float, max_sigma: float
    ) -> List[float]:
        sigmas = [epsilon * max_sigma] * ndim
        sigmas[0] = 0
        sigmas[channel_axis] = 0
        return sigmas

    def _apply_gaussian_filter(self, x: np.ndarray, sigmas: List[float]) -> np.ndarray:
        return gaussian_filter(x, sigmas)

    def _clip_values(self, x: np.ndarray, model: Model) -> np.ndarray:
        min_, max_ = model.bounds
        return np.clip(x, min_, max_)