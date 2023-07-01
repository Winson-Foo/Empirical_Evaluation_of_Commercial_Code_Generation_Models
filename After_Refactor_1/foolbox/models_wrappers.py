import eagerpy as ep

from .base import Model
from .base import T


class ThresholdingWrapper(Model):
    def __init__(self, model: Model, threshold: float):
        self.model = model
        self.threshold = threshold

    @property
    def bounds(self) -> Bounds:
        return self.model.bounds

    def __call__(self, inputs: T) -> T:
        lower_bound, upper_bound = self.model.bounds
        x, restore_type = ep.astensor_(inputs)
        # Apply thresholding
        y = ep.where(x < self.threshold, lower_bound, upper_bound).astype(x.dtype)
        # Pass the modified inputs to the model
        z = self.model(y)
        # Restore the original type of the outputs
        return restore_type(z)