import eagerpy as ep

from your_module.types import Bounds

from your_module.models.base import Model
from your_module.models.base import T

class ThresholdingWrapper(Model):
    def __init__(self, model: Model, threshold: float):
        self._model: Model = model
        self._threshold: float = threshold

    @property
    def bounds(self) -> Bounds:
        return self._model.bounds

    def __call__(self, inputs: T) -> T:
        min_, max_ = self._model.bounds
        x, restore_type = ep.astensor_(inputs)
        y = ep.where(x < self._threshold, min_, max_).astype(x.dtype)
        z = self._model(y)
        return restore_type(z)

def __call__(self, inputs: T) -> T:
    min_bound, max_bound = self._model.bounds
    tensor_input, restore_type = ep.astensor_(inputs)
    thresholded_input = ep.where(tensor_input < self._threshold, min_bound, max_bound).astype(tensor_input.dtype)
    output = self._model(thresholded_input)
    return restore_type(output)