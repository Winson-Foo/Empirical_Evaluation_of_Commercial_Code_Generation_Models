from typing import List, Tuple
import numpy as np

from ...core.node import ProcessorNode
from .constants import BASE_URL_DETECTION


class ObjectDetector(ProcessorNode):
    """
    Abstract class that defines the interface of object detectors.
    """
    def __init__(self) -> None:
        super().__init__()

    def detect(self, image: np.array) -> List[Tuple[float, float, float, float, int, float]]:
        """
        Detects the objects in an image.

        Args:
            image: A numpy array of shape (height, width, 3) representing the input image.

        Returns:
            A list of tuples, each containing the coordinates of a bounding box and its class index
            and score. The box coordinates are returned unnormalized (values NOT between 0 and 1,
            but using the original dimension of the image).
        """
        raise NotImplementedError('Subclass must implement it')

    def process(self, image: np.array) -> List[Tuple[float, float, float, float, int, float]]:
        """
        Processes an input image by detecting objects in it.

        Args:
            image: A numpy array of shape (height, width, 3) representing the input image.

        Returns:
            A list of tuples, each containing the coordinates of a bounding box and its class index
            and score. The box coordinates are returned unnormalized (values NOT between 0 and 1,
            but using the original dimension of the image).
        """
        return self.detect(image)