import eagerpy as ep
from typing import Any, Union

from ..models import Model
from .base import FixedEpsilonAttack
from ..devutils import atleast_kd
from ..distances import l2
from .base import T, raise_if_kwargs, verify_input_bounds

class L2ContrastReductionAttack(FixedEpsilonAttack):
    distance = l2

    def __init__(self, target: float = 0.5):
        self.target = target

    def run(self,
            model: Model,
            inputs: T,
            criterion: Union[Criterion, Any] = None,
            *,
            epsilon: float,
            **kwargs: Any) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)

        verify_input_bounds(x, model)

        min_bound, max_bound = model.bounds
        target_value = min_bound + self.target * (max_bound - min_bound)

        direction = target_value - x
        norms = ep.norms.l2(flatten(direction), axis=-1)
        scale = epsilon / atleast_kd(norms, direction.ndim)
        scale = ep.minimum(scale, 1)

        x = x + scale * direction
        x = x.clip(min_bound, max_bound)
        return restore_type(x)