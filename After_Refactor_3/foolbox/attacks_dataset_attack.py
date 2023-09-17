from typing import Union, Optional, Any, List
import numpy as np
import eagerpy as ep

from foolbox.devutils import atleast_kd
from foolbox.models import Model
from foolbox.distances import Distance
from foolbox.criteria import Criterion
from foolbox.attacks.base import FlexibleDistanceMinimizationAttack
from foolbox.attacks.base import T, get_criterion, raise_if_kwargs, verify_input_bounds


class DatasetAttack(FlexibleDistanceMinimizationAttack):
    """Draws randomly from the given dataset until adversarial examples for all
    inputs have been found.

    To pass data form the dataset to this attack, call :meth:`feed()`.
    :meth:`feed()` can be called several times and should only be called with
    batches that are small enough that they can be passed through the model.

    Args:
        distance : Distance measure for which minimal adversarial examples are searched.
    """

    def __init__(self, *, distance: Optional[Distance] = None):
        super().__init__(distance=distance)
        self.raw_inputs: List[ep.Tensor] = []
        self.raw_outputs: List[ep.Tensor] = []
        self.inputs: Optional[ep.Tensor] = None
        self.outputs: Optional[ep.Tensor] = None

    def feed(self, model: Model, input_data: Any) -> None:
        x = ep.astensor(input_data)
        self.raw_inputs.append(x)
        self.raw_outputs.append(model(x))

    def process_raw(self) -> None:
        raw_inputs = self.raw_inputs
        raw_outputs = self.raw_outputs
        assert len(raw_inputs) == len(raw_outputs)
        assert self.inputs is None == self.outputs is None

        if self.inputs is None:
            if len(raw_inputs) == 0:
                raise ValueError("DatasetAttack can only be called after data has been provided using 'feed()'")
        else:
            raw_inputs = [self.inputs] + raw_inputs
            raw_outputs = [self.outputs] + raw_outputs

        self.inputs = ep.concatenate(raw_inputs, axis=0)
        self.outputs = ep.concatenate(raw_outputs, axis=0)
        self.raw_inputs.clear()
        self.raw_outputs.clear()

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        self.process_raw()
        assert self.inputs is not None
        assert self.outputs is not None
        x = ep.astensor(inputs)
        verify_input_bounds(x, model)
        criterion = get_criterion(criterion)

        result = x
        found = criterion(x, model(x))

        batch_size = len(x)
        index_pools: List[List[int]] = []
        for i in range(batch_size):
            indices = list(range(batch_size))
            indices.remove(i)
            np.random.shuffle(indices)
            index_pools.append(indices)

        for i in range(batch_size - 1):
            if found.all():
                break

            indices = np.array([pool[i] for pool in index_pools])

            xp = self.inputs[indices]
            yp = self.outputs[indices]
            is_adv = criterion(xp, yp)

            new_found = ep.logical_and(is_adv, found.logical_not())
            result = ep.where(atleast_kd(new_found, result.ndim), xp, result)
            found = ep.logical_or(found, new_found)

        return result