from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19

from ISR.utils.logger import get_logger


class CutVGG19:
    """
    Class object that fetches Keras' VGG19 model trained on the ImageNet dataset
    and declares <layers_to_extract> as output layers. Used as a feature extractor
    for the perceptual loss function.
    """

    def __init__(
        self, patch_size: int, layers_to_extract: List[int]
    ) -> None:
        self.PATCH_SIZE: int = patch_size
        self.INPUT_SHAPE: Tuple[int, int, int] = (patch_size, patch_size, 3)
        self.LAYERS_TO_EXTRACT: List[int] = layers_to_extract
        self.logger = get_logger(__name__)

        try:
            self._cut_vgg()
        except:
            self.logger.error("Invalid VGG instantiation: extracted layer must be > 0")
            raise ValueError(
                "Invalid VGG instantiation: extracted layer must be > 0"
            )

    def _cut_vgg(self) -> None:
        """
        Loads pre-trained VGG, declares as the output the intermediate
        layers selected by self.LAYERS_TO_EXTRACT.
        """

        vgg = VGG19(weights="imagenet", include_top=False, input_shape=self.INPUT_SHAPE)
        vgg.trainable = False
        outputs = [vgg.layers[i].output for i in self.LAYERS_TO_EXTRACT]
        self._model = Model([vgg.input], outputs)
        self._model._name = "feature_extractor"
        self._name: str = "vgg19"  # used in weights naming