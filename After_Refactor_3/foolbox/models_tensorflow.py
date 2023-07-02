from typing import cast, Any
import eagerpy as ep
import tensorflow as tf

from ..types import BoundsInput, Preprocessing
from .base import ModelWithPreprocessing


class TensorFlowModel(ModelWithPreprocessing):
    def __init__(
        self,
        model: Any,
        bounds: BoundsInput,
        device: Any = None,
        preprocessing: Preprocessing = None,
    ):
        if not tf.executing_eagerly():
            raise ValueError("TensorFlowModel requires TensorFlow Eager Mode")

        device = tf.device(device) if isinstance(device, str) else device
        with device:
            dummy = ep.tensorflow.zeros(0)
        
        super().__init__(model, bounds, dummy, preprocessing=preprocessing)
        self.device = device

    @property
    def data_format(self) -> str:
        return cast(str, tf.keras.backend.image_data_format())