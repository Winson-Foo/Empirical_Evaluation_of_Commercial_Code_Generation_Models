from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from ...core.node import OneTaskProcessorNode

class BoundingBoxTracker(OneTaskProcessorNode):
    """Tracks bounding boxes from one frame to another.
    Keeps an internal state representation that allows
    tracking across frames.

    Attributes:
        None
    """

    def __init__(self):
        super().__init__()

    def track(self, detections: np.array) -> np.array:
        """Tracks bounding boxes from one frame to another.
        Keeps an internal state representation that allows
        tracking across frames.

        Args:
            detections (np.array): a numpy array of shape (nb_boxes, 6).
                Specifically (nb_boxes, [ymin, xmin, ymax, xmax, class_index, score])

        Returns:
            tracks (np.array): a numpy array of shape (nb_boxes, 5).
                Specifically (nb_boxes, [ymin, xmin, ymax, xmax, track_id])
        """
        raise NotImplementedError("Subclass must implement track method")

    def process(self, detections: np.array) -> np.array:
        """Process the detections and return tracked bounding boxes.

        Args:
            detections (np.array): a numpy array of shape (nb_boxes, 6).
                Specifically (nb_boxes, [ymin, xmin, ymax, xmax, class_index, score])

        Returns:
            np.array: an np array of shape (nb_boxes, 5).
                Specifically (nb_boxes, [ymin, xmin, ymax, xmax, track_id])
        """
        return self.track(detections)