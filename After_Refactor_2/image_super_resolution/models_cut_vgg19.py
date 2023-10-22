import logging

from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19

logger = logging.getLogger(__name__)
LAYER_NAME_ERROR_MSG = "Invalid VGG instantiation: extracted layer must be > 0"


class CutVGG19:
    """
    Object that fetches Keras' VGG19 model trained on the imagenet dataset
    and declares <layers_to_extract> as output layers. Used as feature extractor
    for the perceptual loss function.
    """
    
    def __init__(self, patch_size, layers_to_extract):
        self.patch_size = patch_size
        self.input_shape = (patch_size,) * 2 + (3,)
        self.layers_to_extract = layers_to_extract
        
        if len(self.layers_to_extract) == 0:
            logger.error(LAYER_NAME_ERROR_MSG)
            raise ValueError(LAYER_NAME_ERROR_MSG)
        
        self.feature_extractor = self._cut_vgg()

    def _cut_vgg(self):
        """
        Loads pretrained VGG19, declares as output the intermediate
        layers selected by self.layers_to_extract.
        """
        vgg = VGG19(weights='imagenet', include_top=False, input_shape=self.input_shape)
        vgg.trainable = False
        feature_layers = [vgg.layers[i].output for i in self.layers_to_extract]
        feature_extractor = Model(vgg.input, feature_layers, name='feature_extractor')
        feature_extractor.trainable = False
        return feature_extractor


if __name__ == "__main__":
    vgg = CutVGG19(64, [1, 2, 3])
    print(vgg.feature_extractor.summary())