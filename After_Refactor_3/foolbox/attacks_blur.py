from typing import Union, Optional, Any
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import eagerpy as ep

from ..devutils import atleast_kd
from ..models import Model
from ..criteria import Criterion
from ..distances import Distance
from .base import FlexibleDistanceMinimizationAttack
from .base import T, get_criterion, get_channel_axis, raise_if_kwargs, verify_input_bounds


class GaussianBlurAttack(FlexibleDistanceMinimizationAttack):
    """Blurs the inputs using a Gaussian filter with linearly increasing standard deviation.

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
    ) -> T:
        x, restore_type = ep.astensor_(inputs)
        verify_input_bounds(x, model)

        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)

        if x.ndim != 4:
            raise NotImplementedError(
                "only implemented for inputs with two spatial dimensions (and one channel and one batch dimension)"
            )

        channel_axis = self._get_channel_axis(x.ndim, model)

        max_sigma = self._get_max_sigma(x, channel_axis)

        x0 = x
        x0_ = x0.numpy()

        result = x0
        found = is_adversarial(x0)

        epsilon = 0.0
        stepsize = 1.0 / self.steps
        for _ in range(self.steps):
            epsilon += stepsize
            sigmas = self._calculate_sigmas(epsilon, x.ndim, channel_axis)

            x_ = self._apply_gaussian_filter(x0_, sigmas, model.bounds)
            x = ep.from_numpy(x0, x_)

            is_adv = is_adversarial(x)
            new_adv = ep.logical_and(is_adv, found.logical_not())
            result = ep.where(atleast_kd(new_adv, x.ndim), x, result)
            found = ep.logical_or(new_adv, found)

            if found.all():
                break

        return restore_type(result)

    def _get_channel_axis(self, ndim: int, model: Model) -> int:
        channel_axis = get_channel_axis(model, ndim)
        if self.channel_axis is not None:
            channel_axis = self.channel_axis % ndim

        if channel_axis is None:
            raise ValueError(
                "cannot infer the data_format from the model, please specify"
                " channel_axis when initializing the attack"
            )

        return channel_axis

    def _get_max_sigma(self, x: T, channel_axis: int) -> float:
        max_sigma = self.max_sigma
        if max_sigma is None:
            if channel_axis == 1:
                h, w = x.shape[2:4]
            elif channel_axis == 3:
                h, w = x.shape[1:3]
            else:
                raise ValueError(
                    f"expected 'channel_axis' to be 1 or 3, got {channel_axis}"
                )
            max_sigma = max(h, w)

        return max_sigma

    @staticmethod
    def _calculate_sigmas(epsilon: float, ndim: int, channel_axis: int) -> List[float]:
        sigmas = [epsilon * max_sigma] * ndim
        sigmas[0] = 0
        sigmas[channel_axis] = 0
        return sigmas

    @staticmethod
    def _apply_gaussian_filter(x: np.ndarray, sigmas: List[float], bounds: Tuple[float, float]) -> np.ndarray:
        x_ = gaussian_filter(x, sigmas)
        return np.clip(x_, *bounds)