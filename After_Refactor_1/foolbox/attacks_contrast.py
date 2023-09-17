from typing import Union, Any
import eagerpy as ep

from ..devutils import flatten
from ..distances import l2
from ..models import Model

from .base import FixedEpsilonAttack
from .base import T
from .base import raise_if_kwargs
from .base import verify_input_bounds


class L2ContrastReductionAttack(FixedEpsilonAttack):
    """Reduces the contrast of the input using a perturbation of the given size

    Args:
        target : Target relative to the bounds from 0 (min) to 1 (max)
            towards which the contrast is reduced
    """

    distance = l2

    def __init__(self, *, target: float = 0.5):
        self.target = target

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, Any] = None,
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)

        verify_input_bounds(x, model)

        lower_bound, upper_bound = model.bounds
        target = lower_bound + self.target * (upper_bound - lower_bound)

        direction = target - x
        norms = ep.norms.l2(flatten(direction), axis=-1)
        scale = epsilon / ep.atleast_kd(norms, direction.ndim)
        scale = ep.minimum(scale, 1)

        x = x + scale * direction
        x = x.clip(lower_bound, upper_bound)
        return restore_type(x)