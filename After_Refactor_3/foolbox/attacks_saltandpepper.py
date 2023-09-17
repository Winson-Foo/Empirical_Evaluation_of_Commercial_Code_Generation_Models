from typing import Optional, Any, Union
import eagerpy as ep

from ..criteria import Misclassification
from ..distances import l2
from ..devutils import flatten, atleast_kd
from .base import MinimizationAttack, get_is_adversarial, get_channel_axis, get_criterion
from ..models.base import Model, T
from .base import raise_if_kwargs, verify_input_bounds


class SaltAndPepperNoiseAttack(MinimizationAttack):
    """Increases the amount of salt and pepper noise until the input is misclassified.

    Args:
        steps : The number of steps to run.
        across_channels : Whether the noise should be the same across all channels.
        channel_axis : The axis across which the noise should be the same
            (if across_channels is True). If None, will be automatically inferred
            from the model if possible.
    """

    distance = l2

    def __init__(self, steps: int = 1000, across_channels: bool = True, channel_axis: Optional[int] = None):
        self.steps = steps
        self.across_channels = across_channels
        self.channel_axis = channel_axis

    def run(self, model: Model, inputs: T, criterion: Union[Misclassification, Any], *,
            early_stop: Optional[float] = None, **kwargs: Any) -> T:
        raise_if_kwargs(kwargs)
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        verify_input_bounds(x0, model)

        is_adversarial = get_is_adversarial(criterion_, model)
        N = len(x0)
        shape = list(x0.shape)

        if self.across_channels and x0.ndim > 2:
            channel_axis = get_channel_axis(model, x0.ndim) if self.channel_axis is None else self.channel_axis % x0.ndim
            if channel_axis is not None:
                shape[channel_axis] = 1

        min_, max_ = model.bounds
        r = max_ - min_
        result = x0
        is_adv = is_adversarial(result)
        best_adv_norms = ep.where(is_adv, ep.zeros(x0, N), ep.full(x0, N, ep.inf))
        min_probability = ep.zeros(x0, N)
        max_probability = ep.ones(x0, N)
        stepsizes = max_probability / self.steps
        p = stepsizes

        for step in range(self.steps):
            u = ep.uniform(x0, tuple(shape))
            p_ = atleast_kd(p, x0.ndim)
            salt = (u >= 1 - p_ / 2).astype(x0.dtype) * r
            pepper = -(u < p_ / 2).astype(x0.dtype) * r
            x = x0 + salt + pepper
            x = ep.clip(x, min_, max_)

            norms = flatten(x - x0).norms.l2(axis=-1)
            is_adv = is_adversarial(x)
            is_best_adv = ep.logical_and(is_adv, norms < best_adv_norms)

            result = ep.where(atleast_kd(is_best_adv, x.ndim), x, result)
            best_adv_norms = ep.where(is_best_adv, norms, best_adv_norms)
            min_probability = ep.where(is_best_adv, 0.5 * p, min_probability)
            max_probability = ep.where(is_best_adv, ep.minimum(p * 1.2, 1.0), max_probability)
            remaining = self.steps - step
            stepsizes = ep.where(is_best_adv, (max_probability - min_probability) / remaining, stepsizes)
            reset = p == max_probability
            p = ep.where(ep.logical_or(is_best_adv, reset), min_probability, p)
            p = ep.minimum(p + stepsizes, max_probability)

        return restore_type(result)