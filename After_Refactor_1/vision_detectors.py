from typing import List, Tuple
import numpy as np
from ...core.node import ProcessorNode

DETECTOR_URL_BASE = 'https://github.com/videoflow/videoflow-contrib/releases/download/detector_tf/'

class ObjectDetector(ProcessorNode):
    """Abstract class that defines the interface of object detectors"""

    def __init__(self, model_path: str):
        self.model_path = model_path

    def _detect(self, im: np.ndarray) -> np.ndarray:
        """
        Detect objects in an image
        
        Args:
        im: np.ndarray of shape (h, w, 3)
        
        Returns:
        np.ndarray of shape (nb_boxes, 6)
        Specifically (nb_boxes, [ymin, xmin, ymax, xmax, class_index, score])
        """
        raise NotImplementedError('Subclass must implement it')

    def process(self, im: np.ndarray) -> np.ndarray:
        """
        Process an image and run object detection
        
        Args:
        im: np.ndarray of shape (h, w, 3)
        
        Returns:
        np.ndarray of shape (nb_boxes, 6)
        Specifically (nb_boxes, [ymin, xmin, ymax, xmax, class_index, score])
        The box coordinates are returned unnormalized (values NOT between 0 and 1, but using the original dimension of the image)
        """
        try:
            detections = self._detect(im)
        except Exception as e:
            print(f'Error while processing image: {e}')
            detections = np.empty((0,6))
        return detections

class TFObjectDetector(ObjectDetector):
    """TensorFlow object detector"""

    def __init__(self, model_path: str):
        super().__init__(model_path)

    def _detect(self, im: np.ndarray) -> np.ndarray:
        """
        Detect objects in an image using TensorFlow
        
        Args:
        im: np.ndarray of shape (h, w, 3)
        
        Returns:
        np.ndarray of shape (nb_boxes, 6)
        Specifically (nb_boxes, [ymin, xmin, ymax, xmax, class_index, score])
        """
        # Load model from disk
        model = load_model(self.model_path)
        
        # Preprocess input image
        preprocessed_im = preprocess_image(im)
        
        # Run inference on image
        predictions = model.predict(preprocessed_im)
        
        # Postprocess predictions into bounding boxes
        detections = postprocess_predictions(predictions)
        
        return detections

def preprocess_image(im: np.ndarray) -> np.ndarray:
    """
    Preprocess an image before running inference on it
    
    Args:
    im: np.ndarray of shape (h, w, 3)
    
    Returns:
    np.ndarray of shape (batch_size, h, w, 3)
    """
    # TODO: Implement this function
    pass

def postprocess_predictions(predictions: np.ndarray) -> np.ndarray:
    """
    Postprocess predictions into bounding boxes
    
    Args:
    predictions: np.ndarray of shape (batch_size, nb_anchors, 5 + nb_classes)
    
    Returns:
    np.ndarray of shape (nb_boxes, 6)
    Specifically (nb_boxes, [ymin, xmin, ymax, xmax, class_index, score])
    """
    # TODO: Implement this function
    pass
