from tensorflow.keras.models import Model
from tensorflow.keras.applications import get_model

class FeatureExtractor:
    """
    Class object that fetches a pre-trained model and declares specific layers
    as output layers to be used as feature extractors for the perceptual loss function.

    Args:
        model_name: string, name of the pre-trained model to be used.
        layers_to_extract: list of strings representing layer names to be declared as output layers.
        patch_size: integer, defines the size of the input (patch_size x patch_size).

    Attributes:
        loss_model: multi-output model architecture with <layers_to_extract> as output layers.
        name: string, used in weights naming.
    """
    
    def __init__(self, model_name, layers_to_extract, patch_size):
        self.patch_size = patch_size
        self.input_shape = (patch_size,) * 2 + (3,)
        self.model_name = model_name
        self.layers_to_extract = layers_to_extract
        
        self._load_model()
        self._extract_layers()
        
    def _load_model(self):
        self.model = get_model(self.model_name, weights='imagenet', include_top=False, input_shape=self.input_shape)
        self.model.trainable = False
        self.name = self.model_name.lower()
        
    def _extract_layers(self):
        outputs = [self.model.get_layer(name).output for name in self.layers_to_extract]
        self.loss_model = Model(inputs=self.model.input, outputs=outputs)
        self.loss_model._name = 'feature_extractor'