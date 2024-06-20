from typing import Callable, Optional

import eagerpy as ep

from ..types import Bounds, BoundsInput
from .base import Model


class NumPyModel(Model):
    """
    A model wrapper for NumPy models.
    """

    def __init__(
        self,
        model: Callable,
        bounds: BoundsInput,
        data_format: Optional[str] = None,
    ) -> None:
        """
        Initialize the NumPyModel instance.

        Args:
            model: The NumPy model to be wrapped.
            bounds: The input bounds for the model.
            data_format: The data format of the input (optional).
        """
        self._model = model
        self._bounds = Bounds(*bounds)
        if data_format is not None and data_format not in ["channels_first", "channels_last"]:
            raise ValueError(
                f"expected data_format to be 'channels_first' or 'channels_last', got {data_format}"
            )
        self._data_format = data_format

    @property
    def bounds(self) -> Bounds:
        """
        Get the input bounds for the model.

        Returns:
            The input bounds.
        """
        return self._bounds

    def __call__(self, inputs) -> T:
        """
        Run the model on the given inputs.

        Args:
            inputs: The model inputs.

        Returns:
            The model outputs.
        """
        x, restore_type = ep.astensor_(inputs)
        y = self._model(x.numpy())
        z = ep.from_numpy(x, y)
        return restore_type(z)

    @property
    def data_format(self) -> str:
        """
        Get the data format of the model inputs.

        Returns:
            The data format of the model inputs.

        Raises:
            AttributeError: If data_format was not specified during initialization.
        """
        if self._data_format is None:
            raise AttributeError(
                "please specify data_format when initializing the NumPyModel"
            )
        return self._data_format