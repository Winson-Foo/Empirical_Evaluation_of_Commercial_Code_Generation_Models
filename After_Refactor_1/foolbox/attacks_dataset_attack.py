from typing import Union, Optional, Any, List
import numpy as np
import eagerpy as ep

from ..models import Model
from ..distances import Distance
from ..criteria import Criterion
from .base import FlexibleDistanceMinimizationAttack, T, get_criterion, verify_input_bounds


class DatasetAttack(FlexibleDistanceMinimizationAttack):
    def __init__(self, distance: Optional[Distance] = None):
        super().__init__(distance=distance)
        self.input_batches: List[ep.Tensor] = []
        self.output_batches: List[ep.Tensor] = []
        self.inputs: Optional[ep.Tensor] = None
        self.outputs: Optional[ep.Tensor] = None

    def feed(self, model: Model, inputs: Any) -> None:
        x = ep.astensor(inputs)
        del inputs

        self.input_batches.append(x)
        self.output_batches.append(model(x))

    def process_batches(self) -> None:
        if self.inputs is None:
            if len(self.input_batches) == 0:
                raise ValueError("DatasetAttack can only be called after data has been provided using 'feed()'")
        else:
            assert self.outputs is not None
            self.input_batches = [self.inputs] + self.input_batches
            self.output_batches = [self.outputs] + self.output_batches

        self.inputs = ep.concatenate(self.input_batches, axis=0)
        self.outputs = ep.concatenate(self.output_batches, axis=0)
        self.input_batches = []
        self.output_batches = []

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        self.process_batches()
        x, restore_type = ep.astensor_(inputs)
        del inputs, kwargs

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

            result = ep.where(new_found.unsqueeze(result.ndim), xp, result)
            found = ep.logical_or(found, new_found)

        return restore_type(result)