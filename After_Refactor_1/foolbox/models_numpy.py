import eagerpy as ep

from ..types import Bounds
from ..types import BoundsInput
from .base import Model


class NumPyModel(Model):
    def __init__(self, model, bounds: BoundsInput, data_format: str = None):
        self._model = model
        self._bounds = Bounds(*bounds)
        self._validate_data_format(data_format)
        self._data_format = data_format

    @property
    def bounds(self) -> Bounds:
        return self._bounds

    def __call__(self, inputs):
        x, restore_type = ep.astensor_(inputs)
        y = self._model(x.numpy())
        z = ep.from_numpy(x, y)
        return restore_type(z)

    @property
    def data_format(self) -> str:
        if self._data_format is None:
            raise AttributeError(
                "please specify data_format when initializing the NumPyModel"
            )
        return self._data_format

    def _validate_data_format(self, data_format):
        if data_format is not None and data_format not in ["channels_first", "channels_last"]:
            raise ValueError(
                f"expected data_format to be 'channels_first' or 'channels_last', got {data_format}"
            )