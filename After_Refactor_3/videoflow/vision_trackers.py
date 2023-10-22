from typing import List, Tuple
import numpy as np

class BoundingBoxTracker:
    '''
    Tracks bounding boxes from one frame to another.
    It keeps an internal state representation that allows
    it to track across frames.
    '''

    def track_boxes(self, boxes: np.ndarray) -> np.ndarray:
        '''
        Tracks bounding boxes from one frame to another.

        :param boxes: np.array of shape (nb_boxes, 6) where each row contains 
                      [ymin, xmin, ymax, xmax, class_index, score]
        :return: np.array of shape (nb_boxes, 5) where each row contains 
                 [ymin, xmin, ymax, xmax, track_id]
        '''
        raise NotImplementedError("Subclass must implement _track method")

class OneTaskProcessor:
    '''
    A base class for processors that only have one task to perform.
    '''

    def process(self, inputs: np.ndarray) -> np.ndarray:
        '''
        Processes the inputs and returns the resulting outputs.

        :param inputs: np.array of shape (nb_inputs, input_dimension)
        :return: np.array of shape (nb_outputs, output_dimension)
        '''
        raise NotImplementedError("Subclass must implement process method")

class BoundingBoxProcessor(OneTaskProcessor):
    '''
    A processor that tracks bounding boxes across frames.
    '''

    def __init__(self, tracker: BoundingBoxTracker):
        '''
        Initializes the BoundingBoxProcessor with a BoundingBoxTracker.

        :param tracker: A BoundingBoxTracker instance used to track bounding boxes.
        '''
        self.tracker = tracker

    def process(self, boxes: np.ndarray) -> np.ndarray:
        '''
        Processes the input boxes and returns the tracked boxes.

        :param boxes: np.array of shape (nb_boxes, 6) where each row contains 
                      [ymin, xmin, ymax, xmax, class_index, score]
        :return: np.array of shape (nb_boxes, 5) where each row contains 
                 [ymin, xmin, ymax, xmax, track_id]
        '''
        return self.tracker.track_boxes(boxes)