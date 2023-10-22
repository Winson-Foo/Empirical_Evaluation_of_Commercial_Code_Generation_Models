from typing import List, Tuple
import numpy as np

from ...core.node import ProcessorNode

BASE_URL_DETECTION = 'https://github.com/videoflow/videoflow-contrib/releases/download/detector_tf/'


class ObjectDetector(ProcessorNode):
    '''
    Abstract class that defines the interface of object detectors
    '''
    def detect_objects(self, image: np.array) -> List[Tuple[float]]:
        '''
        Detects objects in an image.

        Arguments:
            image (np.array): An array representing the image, with dimensions (height, width, 3).

        Returns:
            A list of tuples, where each tuple represents an object detected in the image.
            Each tuple contains the following values:
            - ymin (float): The topmost coordinate of the bounding box (in pixels).
            - xmin (float): The leftmost coordinate of the bounding box (in pixels).
            - ymax (float): The bottommost coordinate of the bounding box (in pixels).
            - xmax (float): The rightmost coordinate of the bounding box (in pixels).
            - class_index (int): The index of the class label assigned to the object.
            - score (float): The confidence score assigned to the object detection.
        '''
        raise NotImplementedError('Subclass must implement this method')
    
    def process(self, image: np.array) -> List[Tuple[float]]:
        '''
        Detects objects in an image and returns a list of tuples containing information about each object.

        Arguments:
            image (np.array): An array representing the image, with dimensions (height, width, 3).

        Returns:  
            A list of tuples, where each tuple represents an object detected in the image.
            Each tuple contains the following values:
            - ymin (float): The topmost coordinate of the bounding box (in pixels).
            - xmin (float): The leftmost coordinate of the bounding box (in pixels).
            - ymax (float): The bottommost coordinate of the bounding box (in pixels).
            - xmax (float): The rightmost coordinate of the bounding box (in pixels).
            - class_index (int): The index of the class label assigned to the object.
            - score (float): The confidence score assigned to the object detection.
        '''
        return self.detect_objects(image)