from typing import Any, Optional

from eagerpy import numpy as ep

from ..types import BoundsInput, Preprocessing

from .base import ModelWithPreprocessing


class JAXModel(ModelWithPreprocessing):
    def __init__(
        self,
        model: Any,
        bounds: BoundsInput,
        preprocessing: Preprocessing = None,
        data_format: Optional[str] = "channels_last",
    ):
        dummy = ep.jax.numpy.zeros(0)
        super().__init__(model, bounds=bounds, dummy=dummy, preprocessing=preprocessing)
        self._data_format = data_format

    @property
    def data_format(self) -> str:
        if self._data_format is None:
            msg = "Please specify data_format when initializing the JaxModel"
            raise AttributeError(msg)
        return self._data_format

    def _initialize_dummy(self) -> ep.Tensor:
        dummy = ep.jax.numpy.zeros(0)
        return dummy

    def _initialize_super(
        self, model: Any, bounds: BoundsInput, dummy: ep.Tensor, preprocessing: Preprocessing
    ) -> None:
        super().__init__(model, bounds=bounds, dummy=dummy, preprocessing=preprocessing)

    def _validate_data_format(self) -> None:
        if self._data_format is None:
            msg = "Please specify data_format when initializing the JaxModel"
            raise AttributeError(msg)
        
    def initialize_model(
        self, model: Any, bounds: BoundsInput, preprocessing: Preprocessing
    ) -> None:
        dummy = self._initialize_dummy()
        self._initialize_super(model, bounds, dummy, preprocessing)

    def get_data_format(self) -> str:
        self._validate_data_format()
        return self._data_format