from typing import Union, Any, Optional
import eagerpy as ep

from ..models import Model
from .base import FlexibleDistanceMinimizationAttack
from .base import T
from .base import raise_if_kwargs
from .base import verify_input_bounds


class InversionAttack(FlexibleDistanceMinimizationAttack):
    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Any],
        early_stop: Optional[float] = None,
    ) -> T:
        raise_if_kwargs(kwargs)
        tensor_inputs, restore_type = ep.astensor_(inputs)
        del inputs, criterion, kwargs

        verify_input_bounds(tensor_inputs, model)

        tensor_min, tensor_max = model.bounds
        inverted_tensor = tensor_min + tensor_max - tensor_inputs
        return restore_type(inverted_tensor)