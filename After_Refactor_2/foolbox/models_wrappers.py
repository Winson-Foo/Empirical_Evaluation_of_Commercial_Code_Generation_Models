import eagerpy as ep

from ..types import Bounds
from .base import Model
from .base import T


class ThresholdingWrapper(Model):
    def __init__(self, model: Model, threshold: float):
        self._model = model
        self._threshold = threshold

    @property
    def bounds(self) -> Bounds:
        """Returns the bounds of the wrapped model."""
        return self._model.bounds

    def __call__(self, inputs: T) -> T:
        """Forward pass through the thresholding wrapper."""
        min_bound, max_bound = self._model.bounds
        x, restore_type = ep.astensor_(inputs)
        masked_inputs = self._apply_threshold(x, min_bound, max_bound)
        output = self._model(masked_inputs)
        return restore_type(output)

    def _apply_threshold(self, inputs: T, min_bound: T, max_bound: T) -> T:
        """Applies the thresholding operation to the inputs."""
        below_threshold_mask = inputs < self._threshold
        masked_inputs = ep.where(below_threshold_mask, min_bound, max_bound).astype(inputs.dtype)
        return masked_inputs