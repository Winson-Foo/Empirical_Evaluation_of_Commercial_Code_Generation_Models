from typing import Any, Union

import eagerpy as ep
import tensorflow as tf

from ..types import BoundsInput, Preprocessing
from .base import ModelWithPreprocessing


def get_device(device: Union[str, None]) -> Any:
    """
    Get the TensorFlow device.

    If device is None, use GPU if available, else use CPU.
    If device is a string, create a TensorFlow device using the string.
    """
    if device is None:
        device = tf.device("/GPU:0" if tf.test.is_gpu_available() else "/CPU:0")
    if isinstance(device, str):
        device = tf.device(device)
    return device


class TensorFlowModel(ModelWithPreprocessing):
    """
    Wrapper class for TensorFlow models.
    """

    def __init__(
        self,
        model: Any,
        bounds: BoundsInput,
        device: Any = None,
        preprocessing: Preprocessing = None,
    ):
        """
        Initialize the TensorFlowModel.

        Args:
            model: The TensorFlow model.
            bounds: The input bounds.
            device: The TensorFlow device.
            preprocessing: The preprocessing function.
        """
        if not tf.executing_eagerly():
            raise ValueError("TensorFlowModel requires TensorFlow Eager Mode")

        device = get_device(device)
        with device:
            dummy = ep.astensor([])
        super().__init__(model, bounds, dummy, preprocessing=preprocessing)

        self.device = device

    @property
    def data_format(self) -> str:
        """
        Get the data format of the TensorFlow backend.
        """
        return tf.keras.backend.image_data_format()