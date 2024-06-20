from typing import TypeVar, Callable, Optional

import eagerpy as ep

from ..types import Bounds
from ..types import BoundsInput
from .base import Model


T = TypeVar("T")


class NumPyModel(Model):
    def __init__(
        self,
        model: Callable,
        bounds: BoundsInput,
        data_format: Optional[str] = None,
    ):
        self.model = model
        self.bounds = Bounds(*bounds)
        self.data_format = self._validate_data_format(data_format)

    def __call__(self, inputs: T) -> T:
        x, restore_type = ep.astensor_(inputs)
        y = self.model(x.numpy())
        z = ep.from_numpy(x, y)
        return restore_type(z)

    def _validate_data_format(self, data_format: Optional[str]) -> str:
        valid_formats = ["channels_first", "channels_last"]
        if data_format is None:
            raise AttributeError(
                "Please specify data_format when initializing the NumPyModel."
            )
        if data_format not in valid_formats:
            raise ValueError(
                f"Expected data_format to be one of {valid_formats}, got {data_format}."
            )
        return data_format